import typer

from dr_code import hello


def main(name: str = typer.Option(..., "--name")) -> None:
    typer.echo(hello())
    typer.echo(name)


if __name__ == "__main__":
    typer.run(main)
