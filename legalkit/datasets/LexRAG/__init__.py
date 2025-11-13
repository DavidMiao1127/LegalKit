import os
from typing import List, Dict, Any
from .dataset import LexRAGDataset
from .generator import Generator
from .evaluator import Evaluator
from .retriever import LexRAGRetriever, BM25Config, DenseConfig
from legalkit.storage import RetrievalStorage

__all__ = ["LexRAGDataset", "Generator", "Evaluator", "load_tasks", "maybe_run_retrieval"]

def load_tasks(sub_tasks: List[str] = None):
    """
    Factory to load tasks for verdict_pred using hard-coded relative path.
    """
    ds = LexRAGDataset(sub_tasks)
    return ds.load_data()


def maybe_run_retrieval(args: Dict[str, Any]) -> None:
    """Optionally run retrieval for LexRAG based on CLI/config args.
    Sets env LEGALKIT_LEXRAG_RETRIEVAL_TAG accordingly so dataset loader picks enriched files.
    """
    choice = str(args.get('retrieval_method', 'none')).lower()
    if not choice or choice == 'none':
        return

    # Determine subtasks to run over
    sub_tasks = args.get('sub_tasks')
    mapping = {
        'current_question': 'current_question.jsonl',
        'prefix_question': 'prefix_question.jsonl',
        'prefix_question_answer': 'prefix_question_answer.jsonl',
        'suffix_question': 'suffix_question.jsonl',
    }
    if sub_tasks:
        mapping = {k: v for k, v in mapping.items() if k in set(sub_tasks)}
    if not mapping:
        return

    # Determine run_root to place retrieval artifacts under run_output
    run_root = args.get('run_root') or os.environ.get('LEGALKIT_RUN_ROOT')
    if not run_root:
        # Fallback: do not run retrieval if we cannot determine run directory
        print("[LexRAG] run_root not provided; skipping retrieval stage.")
        return
    storage = RetrievalStorage(run_root, 'LexRAG')
    retriever = LexRAGRetriever(output_dir=storage.base_dir)
    law_file = 'law_library.jsonl'

    tag = None
    if choice in ('bm25s', 'pyserini'):
        tag = f"bm25_{choice}"
        cfg = BM25Config(backend=choice, k=int(args.get('retrieval_k', 10) or 10))
        for _, qf in mapping.items():
            retriever.run_bm25(qf, law_file, cfg)
    elif choice == 'qld':
        tag = 'qld'
        k = int(args.get('retrieval_k', 10) or 10)
        for _, qf in mapping.items():
            retriever.run_qld(qf, law_file, k=k)
    elif choice in ('dense-bge', 'dense-gte', 'dense-api'):
        if choice == 'dense-bge':
            mtype = 'bge-base-zh'
            tag = 'bge'
        elif choice == 'dense-gte':
            mtype = 'gte-qwen2-1.5b'
            tag = 'gte'
        else:
            mtype = 'api'
            # compute tag from model_name or fallback
            mname = args.get('embed_model_name') or 'openai'
            tag = (str(mname)).replace('/', '_')
        dcfg = DenseConfig(
            model_type=mtype,
            model_name=args.get('embed_model_name'),
            api_url=args.get('embed_api_url'),
            api_key=args.get('embed_api_key'),
            batch_size=int(args.get('embed_batch_size', args.get('batch_size', 16)) or 16),
            faiss_type=str(args.get('retrieval_faiss_type', 'FlatIP')),
            k=int(args.get('retrieval_k', 10) or 10),
        )
        for _, qf in mapping.items():
            retriever.run_dense(qf, law_file, dcfg)
    else:
        return

    # Signal dataset loader which retrieval outputs to prefer
    if tag:
        os.environ['LEGALKIT_RETRIEVAL_TAG_LexRAG'] = tag
        os.environ['LEGALKIT_RETRIEVAL_DIR_LexRAG'] = storage.base_dir