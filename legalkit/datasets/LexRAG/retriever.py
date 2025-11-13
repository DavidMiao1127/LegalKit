import os
import json
import math
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# Optional heavy deps guarded at call sites
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

@dataclass
class BM25Config:
    backend: str = "bm25s"  # "bm25s" or "pyserini"
    k: int = 10

@dataclass
class DenseConfig:
    model_type: str  # "bge-base-zh" | "gte-qwen2-1.5b" | "api"
    model_name: Optional[str] = None  # for API (e.g., text-embedding-3-large)
    api_url: Optional[str] = None     # OpenAI-compatible base url
    api_key: Optional[str] = None
    batch_size: int = 16
    faiss_type: str = "FlatIP"        # "FlatIP" | "HNSW" | "IVF"
    k: int = 10

class LexRAGRetriever:
    """
    Unified retriever for LexRAG:
      - Lexical: BM25 (bm25s/pyserini) and QLD
      - Dense: BGE/GTE local embeddings or OpenAI-compatible embeddings via API

    Data layout (relative paths):
      - Questions: data/LexRAG/<preproc>.jsonl
      - Corpus:    data/LexRAG/law_library.jsonl
      - Outputs:   data/LexRAG/retrieval/* (indices, npy, results)
    """

    def __init__(self, output_dir: Optional[str] = None):
        pkg_dir = os.path.dirname(__file__)
        self.dataset_dir = os.path.abspath(os.path.join(pkg_dir, '..', '..', '..', 'data', 'LexRAG'))
        # Where to write retrieval artifacts (indices, npy, results)
        self.retrieval_dir = output_dir or os.path.join(self.dataset_dir, 'retrieval')
        os.makedirs(self.retrieval_dir, exist_ok=True)

    # ----------------------------- Public API ---------------------------------
    def run_bm25(self, question_file: str, law_file: str, cfg: BM25Config) -> str:
        questions, laws, corpus = self._load_questions_and_corpus(question_file, law_file)
        stem = os.path.splitext(os.path.basename(question_file))[0]

        if cfg.backend == "bm25s":
            results, scores = self._bm25s_search(corpus, [q["content"] for q in questions], k=cfg.k)
            out_path = os.path.join(self.retrieval_dir, f"retrieval_{stem}_bm25_{cfg.backend}.jsonl")
            self._write_results_by_conversation(question_file, law_file, results, scores, out_path)
            return out_path
        elif cfg.backend == "pyserini":
            index_dir = os.path.join(self.retrieval_dir, 'pyserini_index')
            self._ensure_pyserini_index(law_file, index_dir)
            docids, scores = self._pyserini_search(index_dir, [q["content"] for q in questions], k=cfg.k)
            out_path = os.path.join(self.retrieval_dir, f"retrieval_{stem}_bm25_{cfg.backend}.jsonl")
            self._write_results_by_conversation(question_file, law_file, docids, scores, out_path, ids_are_numeric=True)
            return out_path
        else:
            raise ValueError(f"Unsupported BM25 backend: {cfg.backend}")

    def run_qld(self, question_file: str, law_file: str, k: int = 10) -> str:
        questions, _, _ = self._load_questions_and_corpus(question_file, law_file)
        stem = os.path.splitext(os.path.basename(question_file))[0]
        index_dir = os.path.join(self.retrieval_dir, 'qld_index')
        self._ensure_pyserini_index(law_file, index_dir)
        docids, scores = self._pyserini_search(index_dir, [q["content"] for q in questions], k=k, method='qld')
        out_path = os.path.join(self.retrieval_dir, f"retrieval_{stem}_qld.jsonl")
        self._write_results_by_conversation(question_file, law_file, docids, scores, out_path, ids_are_numeric=True)
        return out_path

    def run_dense(self, question_file: str, law_file: str, cfg: DenseConfig) -> str:
        if np is None:
            raise RuntimeError("numpy is required for dense retrieval; please `pip install numpy`. ")

        # Paths
        model_label = self._dense_model_label(cfg)
        stem = os.path.splitext(os.path.basename(question_file))[0]
        law_index_path = os.path.join(self.retrieval_dir, f"law_index_{model_label}.faiss")
        q_emb_path = os.path.join(self.retrieval_dir, f"retrieval_{stem}_{model_label}.npy")
        out_path = os.path.join(self.retrieval_dir, f"retrieval_{stem}_{model_label}.jsonl")

        # 1) Build law index if missing
        if not os.path.exists(law_index_path):
            _, laws, corpus = self._load_questions_and_corpus(question_file, law_file)
            law_texts = [l["name"] + l["content"] for l in laws]
            embeds = self._embed_texts(law_texts, cfg)
            self._save_faiss(embeds, cfg.faiss_type, law_index_path)

        # 2) Build question embeddings if missing
        if not os.path.exists(q_emb_path):
            questions, _, _ = self._load_questions_and_corpus(question_file, law_file)
            q_texts = [q["content"] for q in questions]
            q_embeds = self._embed_texts(q_texts, cfg)
            np.save(q_emb_path, q_embeds)

        # 3) Search
        I, D = self._faiss_search(law_index_path, q_emb_path, cfg.k)
        self._write_dense_results(question_file, law_file, I, D, out_path)
        return out_path

    # --------------------------- Data I/O helpers ------------------------------
    def _load_questions_and_corpus(self, question_file: str, law_file: str):
        q_path = os.path.join(self.dataset_dir, question_file) if not os.path.isabs(question_file) else question_file
        l_path = os.path.join(self.dataset_dir, law_file) if not os.path.isabs(law_file) else law_file
        with open(q_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        with open(l_path, 'r', encoding='utf-8') as f:
            laws = [json.loads(line) for line in f]
        # Flatten question turns in order
        questions = [conv["question"] for d in data for conv in d.get("conversation", [])]
        corpus = [law.get("name", "") + law.get("content", "") for law in laws]
        return questions, laws, corpus

    def _write_results_by_conversation(self, question_file: str, law_file: str, result_ids: List[List[int]], scores: List[List[float]], out_path: str, ids_are_numeric: bool = False):
        q_path = os.path.join(self.dataset_dir, question_file) if not os.path.isabs(question_file) else question_file
        l_path = os.path.join(self.dataset_dir, law_file) if not os.path.isabs(law_file) else law_file
        with open(l_path, 'r', encoding='utf-8') as f:
            laws = [json.loads(line) for line in f]
        with open(q_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        idx = 0
        for d in data:
            for conv in d.get("conversation", []):
                tmp = []
                ids = result_ids[idx]
                scs = scores[idx]
                for i, s in zip(ids, scs):
                    law_idx = int(i) if ids_are_numeric else int(i)
                    tmp.append({"article": laws[law_idx], "score": float(s)})
                conv.setdefault("question", {}).setdefault("recall", [])
                conv["question"]["recall"] = tmp
                idx += 1

        with open(out_path, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _write_dense_results(self, question_file: str, law_file: str, I, D, out_path: str):
        q_path = os.path.join(self.dataset_dir, question_file) if not os.path.isabs(question_file) else question_file
        l_path = os.path.join(self.dataset_dir, law_file) if not os.path.isabs(law_file) else law_file
        with open(l_path, 'r', encoding='utf-8') as f:
            laws = [json.loads(line) for line in f]
        with open(q_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        idx = 0
        for d in data:
            for conv in d.get("conversation", []):
                tmp = []
                for j in range(len(I[idx])):
                    tmp.append({"article": laws[int(I[idx][j])], "score": float(D[idx][j])})
                conv.setdefault("question", {}).setdefault("recall", [])
                conv["question"]["recall"] = tmp
                idx += 1

        with open(out_path, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ----------------------------- Lexical impls -------------------------------
    def _bm25s_search(self, corpus: List[str], queries: List[str], k: int) -> Tuple[List[List[int]], List[List[float]]]:
        try:
            import jieba  # type: ignore
            import bm25s  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("bm25s and jieba are required for BM25S backend. Try `pip install bm25s jieba`. ") from e

        # Custom tokenizer: list of token ids
        def _tokenize(texts):
            if isinstance(texts, str):
                texts_ = [texts]
            else:
                texts_ = texts
            corpus_ids = []
            token_to_index: Dict[str, int] = {}
            for text in texts_:
                for token in jieba.lcut(text):
                    if token not in token_to_index:
                        token_to_index[token] = len(token_to_index)
            for text in texts_:
                corpus_ids.append([token_to_index[t] for t in jieba.lcut(text)])
            return bm25s.tokenization.Tokenized(ids=corpus_ids, vocab=token_to_index)

        bm25s.tokenize = _tokenize
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(corpus))

        result_ids: List[List[int]] = []
        scores: List[List[float]] = []
        for q in queries:
            r, s = retriever.retrieve(bm25s.tokenize(q), k=k)
            result_ids.append(r.tolist())
            scores.append(s.tolist())
        return result_ids, scores

    def _ensure_pyserini_index(self, law_file: str, index_dir: str) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyserini is required for BM25/QLD with Lucene backend. Try `pip install pyserini`. ") from e

        if os.path.exists(index_dir) and os.listdir(index_dir):
            return

        folder_path = os.path.join(self.retrieval_dir, 'corpus_for_pyserini')
        os.makedirs(folder_path, exist_ok=True)
        temp_jsonl = os.path.join(folder_path, 'corpus.jsonl')

        # Normalize corpus into 'contents'
        l_path = os.path.join(self.dataset_dir, law_file) if not os.path.isabs(law_file) else law_file
        with open(l_path, 'r', encoding='utf-8') as f_in, open(temp_jsonl, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                data = json.loads(line)
                if 'content' in data:
                    data['contents'] = data.pop('content')
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

        # Detect language from first doc
        with open(temp_jsonl, 'r', encoding='utf-8') as f:
            sample = json.loads(next(f))
            text = sample.get('contents') or sample.get('content') or ''
        language = self._detect_lang(text)

        args = [
            "-collection", "JsonCollection",
            "-input", folder_path,
            "-index", index_dir,
            "-generator", "DefaultLuceneDocumentGenerator",
            "-threads", "1",
        ]
        if language == 'zh':
            args += ["-language", "zh"]

        subprocess.run(["python", "-m", "pyserini.index.lucene"] + args, check=True)

    def _pyserini_search(self, index_dir: str, queries: List[str], k: int = 10, method: str = 'bm25') -> Tuple[List[List[int]], List[List[float]]]:
        from pyserini.search.lucene import LuceneSearcher  # type: ignore
        searcher = LuceneSearcher(index_dir)
        if method == 'qld':
            searcher.set_qld()
        results = []
        scores = []
        for q in queries:
            hits = searcher.search(q, k=k)
            results.append([int(h.docid) for h in hits])
            scores.append([float(h.score) for h in hits])
        return results, scores

    def _detect_lang(self, text: str) -> str:
        try:
            import langid  # type: ignore
            return 'zh' if langid.classify(text)[0] == 'zh' else 'en'
        except Exception:
            # fallback: heuristic
            for ch in text[:64]:
                if '\u4e00' <= ch <= '\u9fff':
                    return 'zh'
            return 'en'

    # ------------------------------ Dense impls --------------------------------
    def _embed_texts(self, texts: List[str], cfg: DenseConfig):
        if cfg.model_type.lower() in ("bge-base-zh", "bge", "bge-base"):
            return self._embed_sentence_transformer(texts, model_name="BAAI/bge-base-zh-v1.5", batch=cfg.batch_size)
        if cfg.model_type.lower() in ("gte-qwen2-1.5b", "gte", "qwen2-gte"):
            return self._embed_sentence_transformer(texts, model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct", batch=cfg.batch_size)
        if cfg.model_type.lower() in ("api", "openai"):
            if not cfg.api_url or not cfg.api_key or not cfg.model_name:
                raise ValueError("For API embeddings, api_url, api_key and model_name must be provided.")
            return self._embed_openai_compatible(texts, api_url=cfg.api_url, api_key=cfg.api_key, model_name=cfg.model_name, batch=cfg.batch_size)
        raise ValueError(f"Unsupported dense model_type: {cfg.model_type}")

    def _embed_sentence_transformer(self, texts: List[str], model_name: str, batch: int):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("sentence-transformers is required for local embeddings. Try `pip install sentence-transformers`. ") from e
        model = SentenceTransformer(model_name, trust_remote_code=True)
        vecs: List[List[float]] = []
        for i in range(0, len(texts), batch):
            vecs.extend(model.encode(texts[i:i+batch]))
        return np.array(vecs)

    def _embed_openai_compatible(self, texts: List[str], api_url: str, api_key: str, model_name: str, batch: int):
        import time
        import requests  # type: ignore
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        out: List[List[float]] = []
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i+batch]
            # Most OpenAI-compatible servers accept batch inputs
            payload = {"model": model_name, "input": batch_texts}
            attempts = 0
            while attempts < 3:
                attempts += 1
                try:
                    resp = requests.post(api_url.rstrip('/') + "/embeddings", json=payload, headers=headers, timeout=60)
                except requests.exceptions.RequestException as e:  # pragma: no cover
                    if attempts < 3:
                        time.sleep(0.8 * attempts)
                        continue
                    raise RuntimeError(f"Embedding API request failed: {e}")
                if resp.status_code != 200:
                    if attempts < 3 and resp.status_code in (429, 500, 502, 503, 504):
                        time.sleep(0.8 * attempts)
                        continue
                    try:
                        err = resp.json()
                    except Exception:
                        err = resp.text
                    raise RuntimeError(f"Embedding API error {resp.status_code}: {err}")
                data = resp.json().get("data", [])
                if not data:
                    if attempts < 3:
                        time.sleep(0.5 * attempts)
                        continue
                    raise RuntimeError("Empty embeddings returned from API")
                # Preserve order
                out.extend([row.get("embedding", []) for row in data])
                break
        return np.array(out)

    def _save_faiss(self, embeddings, faiss_type: str, save_path: str):
        try:
            import faiss  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("faiss is required for dense retrieval; please install faiss-gpu/faiss-cpu.") from e
        dim = embeddings.shape[1]
        if faiss_type == "FlatIP":
            index = faiss.IndexFlatIP(dim)
        elif faiss_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 64)
        elif faiss_type == "IVF":
            nlist = min(128, int(math.sqrt(len(embeddings))))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(embeddings.astype('float32'))
            index.nprobe = max(1, min(8, nlist // 4))
        else:
            raise ValueError(f"Unsupported FAISS type: {faiss_type}")
        index.add(embeddings.astype('float32'))
        faiss.write_index(index, save_path)

    def _faiss_search(self, index_path: str, q_emb_path: str, k: int):
        import faiss  # type: ignore
        index = faiss.read_index(index_path)
        q_emb = np.load(q_emb_path)
        D, I = index.search(q_emb.astype('float32'), k)
        return I, D

    # ------------------------------- utils -------------------------------------
    def _dense_model_label(self, cfg: DenseConfig) -> str:
        t = cfg.model_type.lower()
        if t in ("bge-base-zh", "bge", "bge-base"):
            return "bge"
        if t in ("gte-qwen2-1.5b", "gte", "qwen2-gte"):
            return "gte"
        if t in ("api", "openai"):
            # include model_name safely
            return (cfg.model_name or "openai").replace("/", "_")
        return t
