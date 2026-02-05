import click


@click.command()
@click.option("--input", required=True, help="Input file path")
@click.option("--output", required=True, help="Output file path")
@click.option("--format", type=click.Choice(["json", "colmap", "npz"]), default="json", help="Export format")
def export(input, output, format):
    raise NotImplementedError("Export CLI not yet implemented")
