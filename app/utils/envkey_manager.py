import os


def get_openai_key():
    return os.getenv("OPENAI_API_KEY")


def get_ovh_key():
    return os.getenv("OVHCLOUD_API_KEY")


def get_huggingface_key():
    return os.getenv("HF_TOKEN")


def get_google_key():
    return os.getenv("GOOGLE_API_KEY")


def get_deepseek_key():
    return os.getenv("DEEPSEEK_API_KEY")
