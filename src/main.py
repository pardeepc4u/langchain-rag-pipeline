import click
from src.ingestion.loader import load_documents, split_documents
from src.storage.vectorstore import create_vectorstore_from_chunks
from src.rag.chain import get_qa_chain
from config.settings import settings

@click.group()
def cli():
    """RAG Ollama CLI"""
    pass

@cli.command()
def ingest():
    """Load documents, split, and build the vector database."""
    try:
        docs = load_documents()
        if not docs:
            click.echo(click.style("No documents found.", fg="yellow"))
            return
        
        chunks = split_documents(docs)
        create_vectorstore_from_chunks(chunks)
        click.echo(click.style("✅ Ingestion completed successfully.", fg="green"))
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg="red"))

@cli.command()
@click.argument("query")
def query(query):
    """Ask a question about your documents."""
    try:
        chain = get_qa_chain()
        result = chain({"query": query})
        
        click.echo("\n" + "="*50)
        click.echo(click.style("ANSWER:", fg="cyan"))
        click.echo("-" * 50)
        click.echo(result["result"])
        
        if result.get("source_documents"):
            click.echo("\n" + click.style("SOURCES:", fg="yellow"))
            for i, doc in enumerate(result["source_documents"]):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                content = doc.page_content[:150]
                click.echo(f"\n[Source {i+1}] {source} (Page: {page})")
                click.echo(f"Content: {content}...")
        click.echo("="*50)
    except FileNotFoundError:
        click.echo(click.style("❌ Vector DB not found. Run `rag-ollama ingest` first.", fg="red"))
    except Exception as e:
        click.echo(click.style(f"❌ Error during query: {e}", fg="red"))

if __name__ == "__main__":
    cli()

