import click


@click.command()
@click.option("--name", default="world", help="Who to greet")
def main(name):
    click.echo(click.style(f"Hello, {name}!", fg="green", bold=True))


main()
