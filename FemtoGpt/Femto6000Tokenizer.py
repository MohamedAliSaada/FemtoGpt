import os
import sentencepiece as spm

class Femto6000Tokenizer:
    """
    Self-contained tokenizer wrapper that loads a SentencePiece model
    stored locally in the same folder as this file.
    """

    def __init__(self, tokenizer_name="AR6000SPT"):
        """
        Initializes the tokenizer by loading the .model and .vocab files.

        Args:
            tokenizer_name (str): Base name of the model files (without extension).
                                  Looks for <name>.model and <name>.vocab in the same folder as this file.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))

        self.model_file = os.path.join(base_path, f"{tokenizer_name}.model")
        self.vocab_file = os.path.join(base_path, f"{tokenizer_name}.vocab")

        # Check if files exist
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"Model file not found: {self.model_file}")
        if not os.path.isfile(self.vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_file}")

        # Load the tokenizer model
        self.processor = spm.SentencePieceProcessor(model_file=self.model_file)

    def encode(self, text, out_type='id'):
        """
        Encodes input text into token IDs or subword pieces using SentencePiece.

        Args:
            text (str): The input string to tokenize.
            out_type (str): 'id' (default) for list of token IDs, 'piece' for subword tokens.

        Returns:
            List of token IDs or pieces.
        """
        if out_type == 'id':
            return self.processor.encode(text, out_type=int)  # Calls SentencePieceProcessor.encode to get IDs
        elif out_type == 'piece':
            return self.processor.encode(text, out_type=str)  # Calls SentencePieceProcessor.encode to get pieces
        else:
            raise ValueError("out_type must be 'id' or 'piece'")

    def decode(self, ids_or_pieces):
        """
        Decodes a list of IDs or subword pieces back to text using SentencePiece.

        Args:
            ids_or_pieces (List[int] or List[str]): Tokens to decode.

        Returns:
            str: Decoded natural language string.
        """
        if isinstance(ids_or_pieces[0], int):
            return self.processor.decode(ids_or_pieces)  # Calls SentencePieceProcessor.decode for ID sequence
        elif isinstance(ids_or_pieces[0], str):
            return self.processor.decode_pieces(ids_or_pieces)  # Calls SentencePieceProcessor.decode_pieces for pieces
        else:
            raise ValueError("Input must be list of ints or list of strings")

    def id_to_piece(self, id):
        """Converts a token ID to its subword piece."""
        return self.processor.id_to_piece(id)

    def piece_to_id(self, piece):
        """Converts a subword piece to its token ID."""
        return self.processor.piece_to_id(piece)
