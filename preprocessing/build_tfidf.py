import os
from drqa.build_tfidf import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dbp", "--database_path", type=str, default=None)
    parser.add_argument("-op", "--output_path", type=str, default=None)
    parser.add_argument("-ng", "--ngram", type=int, default=2)
    parser.add_argument("-hs", "--hash_size", type=int, default=int(math.pow(2, 24)))
    parser.add_argument("-t", "--tokenizer", type=str, default="simple")
    parser.add_argument("-n", "--num_workers", type=int, default=None)
    args = parser.parse_args()

    logging.info("Counting words...")
    count_matrix, doc_dict = get_count_matrix(
        args, "sqlite", {"db_path": args.database_path}
    )

    logger.info("Making tfidf vectors...")
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info("Getting word-doc frequencies...")
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.database_path))[0]
    basename += "-tfidf-ngram=%d-hash=%d-tokenizer=%s" % (
        args.ngram,
        args.hash_size,
        args.tokenizer,
    )

    if not os.path.exists(args.output_path):
        logger.info("Creating data directory")
        os.makedirs(args.output_path)

    filename = os.path.join(args.output_path, basename)

    logger.info("Saving to %s.npz" % filename)
    metadata = {
        "doc_freqs": freqs,
        "tokenizer": args.tokenizer,
        "hash_size": args.hash_size,
        "ngram": args.ngram,
        "doc_dict": doc_dict,
    }

    retriever.utils.save_sparse_csr(filename, tfidf, metadata)
