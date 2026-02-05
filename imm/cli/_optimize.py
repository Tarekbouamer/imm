import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model", required=True, help="Model name to export")
@click.option("--output", required=True, help="Output ONNX file path")
@click.option("--opset", default=11, help="ONNX opset version")
def onnx(model, output, opset):
    raise NotImplementedError("ONNX export not yet implemented")


@cli.command()
@click.option("--model", required=True, help="Model name to export")
@click.option("--output", required=True, help="Output TorchScript file path")
def torchscript(model, output):
    raise NotImplementedError("TorchScript export not yet implemented")


@cli.command()
@click.option("--model", required=True, help="Model name to quantize")
@click.option("--output", required=True, help="Output quantized model path")
@click.option("--dtype", type=click.Choice(["int8", "fp16"]), default="int8", help="Quantization dtype")
def quantize(model, output, dtype):
    raise NotImplementedError("Quantization not yet implemented")


@cli.command()
@click.option("--model", required=True, help="Model name to prune")
@click.option("--output", required=True, help="Output pruned model path")
@click.option("--amount", type=float, default=0.3, help="Pruning amount (0.0-1.0)")
def prune(model, output, amount):
    raise NotImplementedError("Model pruning not yet implemented")
