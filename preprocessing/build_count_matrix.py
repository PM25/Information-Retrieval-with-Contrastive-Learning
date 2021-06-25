import os
from drqa.build_tfidf import *
from drqa.retriever.utils import save_sparse_csr
import _pickle as pk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dbp", "--database_path", type=str, default=None)
    parser.add_argument("-op", "--output_path", type=str, default=None)
    parser.add_argument("-ng", "--ngram", type=int, default=2)
    parser.add_argument("-hs", "--hash_size", type=int, default=int(math.pow(2, 24)))
    parser.add_argument("-t", "--tokenizer", type=str, default="simple")
    parser.add_argument("-n", "--num_workers", type=int, default=None)
    args = parser.parse_args()

    count_matrix, doc_dict = get_count_matrix(
            args, "sqlite", {"db_path": args.database_path}
    )

    save_sparse_csr("data/index/count_matrix.npz", count_matrix)