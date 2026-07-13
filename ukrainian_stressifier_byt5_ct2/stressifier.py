import re
import time
from pathlib import Path
import ctranslate2
from huggingface_hub import snapshot_download


class ByT5Tokenizer:
    """
    A lightweight, pure-Python byte-level tokenizer for ByT5 models
    that does not depend on the transformers library.
    """
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def tokenize(self, text: str) -> list[str]:
        # ByT5 maps bytes directly to character tokens (ISO-8859-1 / latin-1 decoding)
        return list(text.encode("utf-8").decode("latin-1"))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        # Filter out special tokens
        clean_tokens = [t for t in tokens if t not in {"<pad>", "</s>", "<unk>"}]
        return "".join(clean_tokens).encode("latin-1").decode("utf-8", errors="replace")


UKRAINIAN_LETTERS = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UKRAINIAN_LETTERS += UKRAINIAN_LETTERS.upper()

VOCAB = UKRAINIAN_LETTERS + " "

UKRAINIAN_RE = re.compile(f"[{UKRAINIAN_LETTERS}]")
UKRAINIAN_RE_DASH = re.compile(f"([{UKRAINIAN_LETTERS}])-([{UKRAINIAN_LETTERS}])")


# --- Text Normalization Utilities ---
def clean_text(text: str) -> str:
    """
    Cleans and normalizes Ukrainian text for processing.
    """
    text = text.lower()
    text = UKRAINIAN_RE_DASH.sub(r"\1 \2", text)
    text = "".join(char for char in text if char in VOCAB)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_and_map(text: str) -> tuple[str, list[int]]:
    """
    Cleans text and builds a map of indices to original positions.
    """
    cleaned = []
    index_map = []
    for i, char in enumerate(text):
        if UKRAINIAN_RE.match(char):
            cleaned.append(char.lower())
            index_map.append(i)
    return "".join(cleaned), index_map


def merge_texts(original: str, with_stress: str) -> str:
    """
    Merges stress annotations from a processed string back into the original.
    """
    norm_original, index_map = clean_and_map(original)
    norm_with_stress = "".join(c for c in with_stress.lower() if c in UKRAINIAN_LETTERS or c == "+")

    stressed_chars = {}
    i = j = 0
    while j < len(norm_with_stress):
        if j < len(norm_with_stress) and norm_with_stress[j] == "+":
            if i > 0:
                stressed_chars[i - 1] += "+"
            j += 1
        elif i < len(norm_original) and norm_with_stress[j] == norm_original[i]:
            stressed_chars[i] = norm_with_stress[j]
            i += 1
            j += 1
        else:
            j += 1

    result = list(original)
    for norm_i, char_index in enumerate(index_map):
        char = result[char_index]
        if norm_i in stressed_chars and "+" in stressed_chars[norm_i]:
            result[char_index] = char + "+"
    return "".join(result)


def split_text_by_whitespace(text: str, max_length: int = 256) -> list[str]:
    """
    Splits long text into chunks with respect to whitespace and a max length.
    """
    words = text.split(" ")
    chunks = []
    current_chunk = None
    for word in words:
        if (
            len("" if current_chunk is None else current_chunk) + len(word) + (1 if current_chunk is not None else 0)
            <= max_length
        ):
            if current_chunk is None:
                current_chunk = word
            else:
                current_chunk += " " + word
        else:
            if current_chunk is not None:
                if chunks:
                    chunks.append(" " + current_chunk)
                else:
                    chunks.append(current_chunk)
            current_chunk = word
    if current_chunk is not None:
        if chunks:
            chunks.append(" " + current_chunk)
        else:
            chunks.append(current_chunk)
    return chunks



def shift_stress_marks_right(text: str, stress_mark="+"):
    text_list = list(text)
    i = 0
    while i <= len(text_list) - 2:
        if text_list[i] == stress_mark:
            text_list[i], text_list[i + 1] = text_list[i + 1], text_list[i]
            i += 1
        i += 1
    return "".join(text_list)


def shift_stress_marks_left(text: str, stress_mark="+"):
    text_list = list(text)
    i = 1
    while i <= len(text_list) - 1:
        if text_list[i] == stress_mark:
            text_list[i - 1], text_list[i] = text_list[i], text_list[i - 1]
            i += 1
        i += 1
    return "".join(text_list)



class UkrainianStressifier:
    """
    Applies stress marks to Ukrainian text using a grapheme-to-phoneme (G2P) model.
    """

    def __init__(
        self,
        model_path: str | None = 'patriotyk/stressifier-byt5-g2p-ctranslate2',
        hf_token: str | None = None,
        max_chunks_length: int | None = 256,
        max_length: int | None  = 256,
        num_beams: int | None  = 2,
        repetition_penalty: float = 1.2,
        device: str = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    ):
        """
        Initializes the stressifier by loading a G2P model and setting decoding parameters.

        Args:
            model_path (Optional[str]): Path to a local model directory or HF repo.
            hf_token (Optional[str]): Hugging Face token for downloading the model.
            max_chunks_length (Optional[int]): Max chunk length for internal splitting.
            max_length (Optional[int]): Maximum generation length for the model.
            num_beams (Optional[int]): Number of beams for beam search decoding.
        """
        self.device = device
        self.max_chunks_length = max_chunks_length
        self.max_length = max_length
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty

        # Resolve model path: check if it is a local directory containing model.bin
        if model_path and Path(model_path).is_dir() and (Path(model_path) / "model.bin").exists():
            local_path = model_path
        else:
            local_path = snapshot_download(repo_id=model_path, token=hf_token)

        self.tokenizer = ByT5Tokenizer.from_pretrained(local_path, token=hf_token)

        self.translator = ctranslate2.Translator(local_path, device=self.device)

    def apply_stress_marks(self, text: str, stress_after_vowel: bool = True) -> str:
        """
        Inserts stress marks into Ukrainian text.
        """
        chunks = split_text_by_whitespace(text, max_length=self.max_chunks_length)
        result_chunks = []

        for chunk in chunks:
            cleaned = clean_text(chunk)
            match = re.search(rf"[{UKRAINIAN_LETTERS}]", cleaned)

            if not match:
                result_chunks.append(chunk)
                continue

            prefix = cleaned[:match.start()]
            core = cleaned[match.start():]
            ends_with_period = core.endswith(".")

            if not ends_with_period:
                core += "."

            
            source_tokens = self.tokenizer.tokenize(core)
            #start = time.time()
            results = self.translator.translate_batch(
                [source_tokens],
                max_decoding_length=self.max_length,
                beam_size=self.num_beams,
                repetition_penalty=self.repetition_penalty
            )
            end = time.time()
            #print(f"Translate time: {end - start}")
            output_tokens = results[0].hypotheses[0]
            stressed = self.tokenizer.convert_tokens_to_string(output_tokens)            
            stressed = shift_stress_marks_right(stressed)

            if stressed.endswith(".") and not ends_with_period:
                stressed = stressed[:-1]

            merged = merge_texts(chunk, prefix + stressed)
            result_chunks.append(merged)

        result = "".join(result_chunks)
        if not stress_after_vowel:
            result = shift_stress_marks_left(result)
        return result


if __name__ == "__main__":
    stressifier = UkrainianStressifier()
    text = 'Привіт! (це тест, чи не так?) - так, дійсно...'
    result = stressifier.apply_stress_marks(text)
    print(result)
