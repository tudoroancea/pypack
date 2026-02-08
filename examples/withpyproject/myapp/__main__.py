import click


@click.command()
@click.option("--name", default="world")
def main(name):
    click.echo(f"Hello, {name}!")


main()
