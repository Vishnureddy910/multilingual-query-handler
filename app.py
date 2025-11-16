import streamlit as st
from langdetect import detect
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sentence_transformers import SentenceTransformer, util

# ----------------- MBART language codes -----------------
MBART_LANG_CODES = {
    "ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX",
    "et": "et_EE", "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN",
    "hr": "hr_HR", "hu": "hu_HU", "id": "id_ID", "it": "it_IT", "ja": "ja_XX",
    "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT", "lv": "lv_LV", "ml": "ml_IN",
    "mr": "mr_IN", "ne": "ne_NP", "nl": "nl_XX", "ro": "ro_RO", "ru": "ru_RU",
    "si": "si_LK", "sv": "sv_SE", "ta": "ta_IN", "te": "te_IN", "th": "th_TH",
    "tr": "tr_TR", "uk": "uk_UA", "ur": "ur_PK", "vi": "vi_VN", "zh-cn": "zh_CN",
    "zh-tw": "zh_TW",
}

# ----------------- LOAD MODELS -----------------------
@st.cache_resource
def load_translation_pipeline():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
    return tokenizer, model

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tokenizer, translation_model = load_translation_pipeline()
embedder = load_embedder()

# ---------------- STREAMLIT UI -----------------------
st.title("🌐 Real-Time Multilingual Batch Translator (All Languages)")

st.markdown("""
Enter multiple texts, **one per line**, in any language.  
The app will translate each line to English and show embedding similarity.
""")
user_input = st.text_area("Enter texts here:", height=250)
submit = st.button("Translate & Analyze")

# ---------------- FUNCTIONS --------------------------
def translate_to_english(text):
    try:
        lang = detect(text).lower()
    except:
        lang = "en"
    tokenizer.src_lang = MBART_LANG_CODES.get(lang, "en_XX")
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# ---------------- OUTPUT -----------------------------
if submit and user_input.strip():
    st.info("Processing your texts...")

    lines = user_input.strip().split("\n")
    for i, line in enumerate(lines):
        st.markdown(f"### Text {i+1}")
        st.write(f"**Original:** {line}")

        # Translate
        english = translate_to_english(line)
        st.success(f"**Translated:** {english}")

        # Embedding similarity (optional)
        original_emb = embedder.encode(line, convert_to_tensor=True)
        english_emb = embedder.encode(english, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(original_emb, english_emb).item()
        st.write(f"Embedding similarity score: {similarity:.4f}")
