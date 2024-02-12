from takpy import new_game, Game, MoveKind, Piece


def generate_openings(game: Game, depth: int, games: set[Game]):
    if depth <= 0:
        games.add(game.canonical())
        return
    for move in game.possible_moves():
        if move.kind != MoveKind.Place and move.piece != Piece.Flat:
            continue
        generate_openings(game.clone_and_play(move), depth - 1, games)


if __name__ == "__main__":
    openings = set()
    generate_openings(new_game(size=4, half_komi=4), 3, openings)
    for game in openings:
        print(game)
