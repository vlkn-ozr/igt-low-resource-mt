# Thesis Repository: Leveraging Interlinear Glossed Text for Low-Resource Machine Translation

## 📁 Directory Structure

### Data Processing & Dictionary Building

#### [`awesome_align/`](awesome_align/)
Word alignment tools using Awesome Align for creating aligned dictionaries from parallel sentences.

- **Purpose**: Extract word-level alignments between Turkish and English parallel sentences
- **See**: [README.md](awesome_align/README.md)

#### [`build_dictionary/`](build_dictionary/)
Scripts for processing SETIMES parallel corpus and creating Turkish-English aligned dictionaries.

- **Purpose**: Build dictionaries from aligned parallel corpora with filtering options
- **See**: [README.md](build_dictionary/README.md)

#### [`lemma_mapping/`](lemma_mapping/)
Tools for processing Turkish-English parallel data, creating dictionaries, and translating glosses.

- **Purpose**: Translate Turkish glosses to English using dictionaries, handle lemmatization
- **See**: [README.md](lemma_mapping/README.md)

### GlossLM Dataset Processing

#### [`glosslm_gloss_generation/`](glosslm_gloss_generation/)
Evaluation scripts for GlossLM model gloss generation.

- **Purpose**: Evaluate GlossLM model performance on test sets
- **See**: [README.md](glosslm_gloss_generation/README.md)

#### [`glosslm_llm_finetune_data/`](glosslm_llm_finetune_data/)
Scripts for processing GlossLM dataset for LLM fine-tuning.

- **Purpose**: Download, filter, and prepare GlossLM data for Qwen model fine-tuning
- **See**: [README.md](glosslm_llm_finetune_data/README.md)

#### [`glosslm_nmt_subset/`](glosslm_nmt_subset/)
Processing scripts for creating NMT training subsets from GlossLM.

- **Purpose**: Extract and prepare GlossLM data for neural machine translation training
- **See**: [README.md](glosslm_nmt_subset/README.md)

#### [`shared_task_lang_glosslm/`](shared_task_lang_glosslm/)
Extraction scripts for out-of-domain languages from GlossLM corpus.

- **Purpose**: Extract data for Gitksan, Lezgi, Natugu, and Nyangbo languages
- **See**: [README.md](shared_task_lang_glosslm/README.md)

### Language Model Fine-tuning

#### [`llm_finetune/`](llm_finetune/)
Fine-tuning Qwen models (Qwen2.5-7B-Instruct and Qwen3-8B) for linguistic gloss translation.

- **Purpose**: Fine-tune large language models to translate glosses to natural English
- **See**: [README.md](llm_finetune/README.md)

#### [`llm_dict_finetune/`](llm_dict_finetune/)
Fine-tuning Qwen2.5-7B-Instruct for Turkish-English word alignment.

- **Purpose**: Train LLM to generate word alignments between parallel sentences
- **See**: [README.md](llm_dict_finetune/README.md)

#### [`llm_retrieval/`](llm_retrieval/)
Retrieval-based approaches for gloss translation and dictionary building.

- **Purpose**: Implement retrieval mechanisms for few-shot learning and dictionary lookup
- **See**: [README.md](llm_retrieval/README.md)

#### [`llm_dict_assesment/`](llm_dict_assesment/)
Assessment and evaluation tools for dictionary quality.

- **Purpose**: Evaluate dictionary quality, coverage, and translation accuracy
- **See**: [README.md](llm_dict_assesment/README.md)

### Neural Machine Translation

#### [`nmt_training/`](nmt_training/)
Neural machine translation training with BPE tokenization using OpenNMT-py.

- **Purpose**: Train NMT models for gloss-to-translation and transcription-to-translation tasks
- **See**: [README.md](nmt_training/README.md)

### Morphological Analysis

#### [`morph_analyzers/`](morph_analyzers/)
Turkish morphological analyzers including Google Morph, TRMOR, and TRmorph.

- **Purpose**: Morphological analysis and disambiguation for Turkish text
- **Subdirectories**:
  - `google-morph/`: Google's Turkish morphology analyzer
  - `TRMOR/`: TRMOR morphological analyzer
  - `TRmorph/`: TRmorph morphological analyzer

#### [`simplify_trmorph_labels/`](simplify_trmorph_labels/)
Scripts for simplifying TRmorph morphological labels.

- **Purpose**: Reduce complexity of morphological gloss labels
- **See**: [README.md](simplify_trmorph_labels/README.md)

#### [`trmorph_glosslm_gloss_mapping/`](trmorph_glosslm_gloss_mapping/)
Mapping between TRmorph glosses and GlossLM gloss format.

- **Purpose**: Convert between different gloss annotation formats
- **See**: [README.md](trmorph_glosslm_gloss_mapping/README.md)

### Fieldwork Data Processing

#### [`wav2gloss_fieldwork/`](wav2gloss_fieldwork/)
Processing scripts for wav2gloss/fieldwork dataset from Hugging Face.

- **Purpose**: Download, extract, and analyze fieldwork language data
- **See**: [README.md](wav2gloss_fieldwork/README.md)

## 🛠️ Requirements

Most directories have their own `requirements.txt` files. Common dependencies include:

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- OpenNMT-py (for NMT)
- SentencePiece (for BPE)
- pandas, numpy
- huggingface_hub

Install dependencies per directory:
```bash
cd <directory>
pip install -r requirements.txt
```

## 🤗 Fine-tuned Models

The fine-tuned language models developed in this thesis are publicly available on **Hugging Face**:

- **Qwen2.5-7B (Turkish)**: [`vlkn/FT-Qwen2.5-7B-TR`](https://huggingface.co/vlkn/FT-Qwen2.5-7B-TR)
- **Qwen3-8B (Turkish)**: [`vlkn/FT-Qwen3-8B-TR`](https://huggingface.co/vlkn/FT-Qwen3-8B-TR)
- **Qwen2.5-7B (GlossLM fine-tuned)**: [`vlkn/FT-Qwen2.5-7B-GlossLM`](https://huggingface.co/vlkn/FT-Qwen2.5-7B-GlossLM)

## 🔗 External Resources

- **GlossLM**: [Hugging Face Dataset](https://huggingface.co/lecslab/glosslm)
- **SETIMES Corpus**: [OPUS](http://opus.nlpl.eu/SETIMES-v2.php)
- **Awesome Align**: [GitHub](https://github.com/neulab/awesome-align)
- **OpenNMT-py**: [GitHub](https://github.com/OpenNMT/OpenNMT-py)
