from app.utils.logger_manager import setup_logger
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Levenshtein import distance
from sentence_transformers import SentenceTransformer, util


logger = setup_logger(__package__, __file__)


def bleu_score(output, expected_output):
    smooth = SmoothingFunction().method1
    return sentence_bleu(
        [expected_output.split()],
        output.split(),
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )


def rouge_scores(output, expected_output, rouge_score="rouge-l"):
    return Rouge().get_scores(output, expected_output, avg=True)[rouge_score]["f"]


def levenshtein_evaluation(output, expected_output):
    score = distance(output, expected_output) / max(len(output), len(expected_output))
    logger.info(f"Levenshtein Score: {score}")
    return score


def cosine_evaluation(output, expected_output, cosine_model):
    model = SentenceTransformer(cosine_model)
    emb1 = model.encode(output, convert_to_tensor=True)
    emb2 = model.encode(expected_output, convert_to_tensor=True)

    similarity = float(util.cos_sim(emb1, emb2)[0][0])
    normalized_score = (similarity + 1) / 2
    logger.info(f"Cosine Score: {normalized_score}")

    return normalized_score


def exact_match(output, expected_output):
    return 1.0 if output == expected_output else 0.0
