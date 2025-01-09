class KVPressMultiTurnPipeline(Pipeline):
    """
    Pipeline for multi-turn conversations with key-value compression in causal language models.
    Supports both single-turn and multi-turn scenarios.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
        self.total_sequence_length = 0
        self.persistent_cache = None

    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        cache: Optional[Cache] = None,
        is_continuation: bool = False,
        **kwargs,
    ):
        """
        Sanitize the input parameters for the pipeline.
        Extended version of the original sanitization that adds multi-turn support.
        """
        answer_prefix = answer_prefix or ""
        postprocess_kwargs = {"single_question": questions is None}
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        questions = questions or ([question] if question else [""])
        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))
            
        preprocess_kwargs = {
            "questions": questions,
            "answer_prefix": answer_prefix,
            "max_context_length": max_context_length,
        }
        forward_kwargs = {
            "press": press, 
            "max_new_tokens": max_new_tokens, 
            "cache": cache,
            "is_continuation": is_continuation  # Added parameter for multi-turn
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
    ):
        """
        Apply the chat template to the context and questions, handling conversation history.
        """
        # Handle conversation history if it exists
        if self.conversation_history:
            context = "\n".join(self.conversation_history + [context])

        # Original chat template handling
        if self.tokenizer.chat_template is None:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context = bos_token + context
            question_suffix = "\n"
        else:
            separator = "\n" + "#" * len(context)
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}], 
                add_generation_prompt=True, 
                tokenize=False
            )
            context, question_suffix = context.split(separator)

        questions = [question + question_suffix + answer_prefix for question in questions]

        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) 
            for question in questions
        ]

        if context_ids.shape[1] > max_context_length:
            logger.warning(
                f"Context length has been truncated from {context_ids.shape[1]} to {max_context_length} tokens."
            )
            context_ids = context_ids[:, :max_context_length]

        return {"context_ids": context_ids, "questions_ids": question_ids}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
        is_continuation: bool = False,
    ):
        """
        Forward pass handling both single-turn and multi-turn scenarios.
        """
        # Use persistent cache for continuation, or initialize new one
        if is_continuation and self.persistent_cache is not None:
            cache = self.persistent_cache
        else:
            self.persistent_cache = DynamicCache() if cache is None else cache
            cache = self.persistent_cache

        context_ids = input_tensors["context_ids"].to(self.model.device)
        
        # Process context only for new conversations or empty cache
        if not is_continuation or cache.get_seq_length() == 0:
            with press(self.model) if press is not None else contextlib.nullcontext():
                self.model(
                    input_ids=context_ids,
                    past_key_values=cache,
                    output_attentions=self.output_attentions(press),
                    num_logits_to_keep=1,
                )
            self.total_sequence_length = cache.get_seq_length()

        answers = []
        for question_ids in input_tensors["questions_ids"]:
            answer = self.generate_answer(
                question_ids=question_ids.to(self.model.device),
                cache=cache,
                context_length=self.total_sequence_length,
                max_new_tokens=max_new_tokens,
            )
            answers.append(answer)
            self.conversation_history.append(answer)

        return answers

    def output_attentions(self, press: BasePress):
        """Maintained from original implementation"""
        if isinstance(press, ObservedAttentionPress):
            return True
        if isinstance(press, (KeyRerotationPress, PerLayerCompressionPress)) and isinstance(
            press.press, ObservedAttentionPress
        ):
            return True
        return False

    def generate_answer(
        self, 
        question_ids: torch.Tensor, 
        cache: Cache, 
        context_length: int, 
        max_new_tokens: int
    ) -> str:
        """
        Generate an answer, handling position IDs for multi-turn conversations.
        """
        cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
        
        position_ids = torch.arange(
            self.total_sequence_length,
            self.total_sequence_length + question_ids.shape[1],
            device=self.model.device
        ).unsqueeze(0)

        outputs = self.model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )

        self.total_sequence_length += question_ids.shape[1]
        
        # Original generation logic maintained
        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            outputs = self.model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break

        return self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)

    def reset_conversation(self):
        """Reset the conversation state"""
        self.conversation_history = []
        self.total_sequence_length = 0
        self.persistent_cache = None

    def postprocess(self, model_outputs, single_question):
        """Maintained from original implementation"""
        if single_question:
            return {"answer": model_outputs[0]}
        return {"answers": model_outputs}
