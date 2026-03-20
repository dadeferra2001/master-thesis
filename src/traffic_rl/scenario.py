"""2x2 scenario metadata and traffic movement definitions."""

from __future__ import annotations

from dataclasses import dataclass


TLS_IDS = ("1", "2", "5", "6")


@dataclass(frozen=True)
class Origin:
    name: str
    edge: str
    heading: str
    row: int
    col: int


WEST_OUT_BY_ROW = {0: "h11", 1: "h21"}
EAST_OUT_BY_ROW = {0: "-h13", 1: "-h23"}
NORTH_OUT_BY_COL = {0: "v11", 1: "v21"}
SOUTH_OUT_BY_COL = {0: "-v13", 1: "-v23"}


ORIGINS = (
    Origin("west_north", "-h11", "east", row=0, col=0),
    Origin("west_south", "-h21", "east", row=1, col=0),
    Origin("east_north", "h13", "west", row=0, col=1),
    Origin("east_south", "h23", "west", row=1, col=1),
    Origin("north_west", "-v11", "south", row=0, col=0),
    Origin("north_east", "-v21", "south", row=0, col=1),
    Origin("south_west", "v13", "north", row=1, col=0),
    Origin("south_east", "v23", "north", row=1, col=1),
)


def turn_destinations(origin: Origin) -> dict[str, str]:
    if origin.heading == "east":
        return {
            "straight": EAST_OUT_BY_ROW[origin.row],
            "left": NORTH_OUT_BY_COL[origin.col],
            "right": SOUTH_OUT_BY_COL[origin.col],
        }
    if origin.heading == "west":
        return {
            "straight": WEST_OUT_BY_ROW[origin.row],
            "left": SOUTH_OUT_BY_COL[origin.col],
            "right": NORTH_OUT_BY_COL[origin.col],
        }
    if origin.heading == "south":
        return {
            "straight": SOUTH_OUT_BY_COL[origin.col],
            "left": EAST_OUT_BY_ROW[origin.row],
            "right": WEST_OUT_BY_ROW[origin.row],
        }
    if origin.heading == "north":
        return {
            "straight": NORTH_OUT_BY_COL[origin.col],
            "left": WEST_OUT_BY_ROW[origin.row],
            "right": EAST_OUT_BY_ROW[origin.row],
        }
    raise ValueError(f"Unsupported heading: {origin.heading}")


def agent_index(agent_id: str) -> int:
    return TLS_IDS.index(agent_id)
