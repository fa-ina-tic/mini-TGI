"""Command-line interface for Mini-TGI."""

import argparse


def cli():
    """Main CLI entry point for Mini-TGI."""
    parser = argparse.ArgumentParser(
        description="Mini-TGI: A lightweight inference server with OpenAI API compatibility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the Mini-TGI server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server options
    server_group = serve_parser.add_argument_group("Server options")
    server_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )

    # Generator options
    gen_group = serve_parser.add_argument_group("Generator options")
    gen_group.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model ID to load (can include revision as model_id@revision)",
    )
    gen_group.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    gen_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from the model repository",
    )
    gen_group.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use",
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.command == "serve":
        from .api import serve

        serve(
            model_id=args.model_id,
            host=args.host,
            port=args.port,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
            seed=args.seed,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
