"""
GraphQL query templates for the Polymarket subgraphs on The Graph.

Three subgraphs are relevant:

* **activity** – Splits, Merges, Redemptions, Positions, Conditions.
* **orderbook** – OrderFilledEvent, OrdersMatchedEvent, Orderbook, MarketData.
* **pnl** – UserPosition, Condition (with payout numerators).

Each query constant is a (query_string, default_variables) tuple so the
subgraph client can merge caller-provided overrides into the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Default subgraph IDs (can be overridden via env)
# ---------------------------------------------------------------------------

# Main activity subgraph published on The Graph Network
ACTIVITY_SUBGRAPH_ID = "Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"

# These may need to be looked up on The Graph Explorer; placeholders until
# confirmed.  Set via ORDERBOOK_SUBGRAPH_ID / PNL_SUBGRAPH_ID env vars.
ORDERBOOK_SUBGRAPH_ID: str | None = None
PNL_SUBGRAPH_ID: str | None = None


# ---------------------------------------------------------------------------
# Query dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubgraphQuery:
    """Immutable container for a named GraphQL query."""
    name: str
    query: str
    default_variables: dict[str, Any] = field(default_factory=dict)
    subgraph: str = "activity"          # activity | orderbook | pnl
    description: str = ""
    response_key: str | None = None


# ===================================================================
# Activity subgraph queries
# ===================================================================

MARKETS = SubgraphQuery(
    name="markets",
    subgraph="activity",
    description="Fetch Polymarket markets with question, slug, and token IDs.",
    query="""\
query Markets($first: Int!, $skip: Int!) {
  markets(
    first: $first
    skip: $skip
    orderBy: createdAt
    orderDirection: asc
  ) {
    id
    conditionId
    question
    slug
    outcomeTokenIds
    resolved
    resolvedTime
    createdAt
  }
}
""",
    default_variables={"first": 1000, "skip": 0},
)

CONDITIONS = SubgraphQuery(
    name="conditions",
    subgraph="activity",
    description="Fetch condition IDs (market identifiers).",
    query="""\
query Conditions($first: Int!, $skip: Int!) {
  conditions(
    first: $first
    skip: $skip
    orderBy: id
    orderDirection: asc
  ) {
    id
  }
}
""",
    default_variables={"first": 1000, "skip": 0},
)

POSITIONS = SubgraphQuery(
    name="positions",
    subgraph="activity",
    description="Fetch positions (ERC-1155 token → condition mapping).",
    query="""\
query Positions($first: Int!, $skip: Int!) {
  positions(
    first: $first
    skip: $skip
    orderBy: id
    orderDirection: asc
  ) {
    id
    condition
    outcomeIndex
  }
}
""",
    default_variables={"first": 1000, "skip": 0},
)

POSITIONS_BY_TOKEN_IDS = SubgraphQuery(
    name="positionsByTokenIds",
    subgraph="activity",
    description="Fetch positions filtered by token IDs.",
    query="""\
query PositionsByTokenIds($first: Int!, $skip: Int!, $tokenIds: [String!]) {
  positions(
    first: $first
    skip: $skip
    orderBy: id
    orderDirection: asc
    where: { id_in: $tokenIds }
  ) {
    id
    condition
    outcomeIndex
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "tokenIds": []},
)

REDEMPTIONS = SubgraphQuery(
    name="redemptions",
    subgraph="activity",
    description="Fetch redemption events (settlement / resolution proof).",
    query="""\
query Redemptions($first: Int!, $skip: Int!, $since: Int!) {
  redemptions(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
    where: { timestamp_gt: $since }
  ) {
    id
    timestamp
    redeemer
    condition
    indexSets
    payout
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "since": 0},
)

SPLITS = SubgraphQuery(
    name="splits",
    subgraph="activity",
    description="Fetch split events (collateral → outcome tokens).",
    query="""\
query Splits($first: Int!, $skip: Int!, $since: Int!) {
  splits(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
    where: { timestamp_gt: $since }
  ) {
    id
    timestamp
    stakeholder
    condition
    amount
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "since": 0},
)

MERGES = SubgraphQuery(
    name="merges",
    subgraph="activity",
    description="Fetch merge events (outcome tokens → collateral).",
    query="""\
query Merges($first: Int!, $skip: Int!, $since: Int!) {
  merges(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
    where: { timestamp_gt: $since }
  ) {
    id
    timestamp
    stakeholder
    condition
    amount
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "since": 0},
)

NEG_RISK_EVENTS = SubgraphQuery(
    name="negRiskEvents",
    subgraph="activity",
    description="Fetch NegRisk event metadata.",
    query="""\
query NegRiskEvents($first: Int!, $skip: Int!) {
  negRiskEvents(
    first: $first
    skip: $skip
    orderBy: id
    orderDirection: asc
  ) {
    id
    questionCount
  }
}
""",
    default_variables={"first": 1000, "skip": 0},
)

NEG_RISK_CONVERSIONS = SubgraphQuery(
    name="negRiskConversions",
    subgraph="activity",
    description="Fetch NegRisk conversion events.",
    query="""\
query NegRiskConversions($first: Int!, $skip: Int!, $since: Int!) {
  negRiskConversions(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
    where: { timestamp_gt: $since }
  ) {
    id
    timestamp
    stakeholder
    negRiskMarketId
    amount
    indexSet
    questionCount
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "since": 0},
)


# ===================================================================
# Orderbook subgraph queries
# ===================================================================

TRADES = SubgraphQuery(
    name="trades",
    subgraph="orderbook",
    description="Fetch trade fills (marketId/outcomeTokenId/price/size).",
    query="""\
query Trades($first: Int!, $skip: Int!, $since: Int!) {
  trades(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
    where: { timestamp_gt: $since }
  ) {
    id
    marketId
    outcomeTokenId
    price
    size
    side
    timestamp
    blockNumber
    transactionHash
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "since": 0},
)

TRADES_BY_MARKET = SubgraphQuery(
    name="tradesByMarket",
    subgraph="orderbook",
    description="Fetch trade fills filtered by market IDs.",
    response_key="trades",
    query="""\
query TradesByMarket($first: Int!, $skip: Int!, $since: Int!, $marketIds: [String!]) {
  trades(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
    where: { timestamp_gt: $since, marketId_in: $marketIds }
  ) {
    id
    marketId
    outcomeTokenId
    price
    size
    side
    timestamp
    blockNumber
    transactionHash
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "since": 0, "marketIds": []},
)

ORDER_FILLED_EVENTS = SubgraphQuery(
    name="orderFilledEvents",
    subgraph="orderbook",
    description=(
        "Fetch individual order fills — the primary trade-level data. "
        "Each fill records maker/taker, asset IDs, amounts, fee, and timestamp."
    ),
    query="""\
query OrderFilledEvents($first: Int!, $skip: Int!, $since: Int!) {
  orderFilledEvents(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
    where: { timestamp_gt: $since }
  ) {
    id
    transactionHash
    timestamp
    orderHash
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
    fee
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "since": 0},
)

ORDERS_MATCHED_EVENTS = SubgraphQuery(
    name="ordersMatchedEvents",
    subgraph="orderbook",
    description="Fetch matched-order events (pairs of fills).",
    query="""\
query OrdersMatchedEvents($first: Int!, $skip: Int!, $since: Int!) {
  ordersMatchedEvents(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
    where: { timestamp_gt: $since }
  ) {
    id
    timestamp
    makerAssetID
    takerAssetID
    makerAmountFilled
    takerAmountFilled
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "since": 0},
)

ORDERBOOK_STATS = SubgraphQuery(
    name="orderbooks",
    subgraph="orderbook",
    description="Fetch per-token aggregate orderbook statistics (volume, trade count).",
    query="""\
query Orderbooks($first: Int!, $skip: Int!) {
  orderbooks(
    first: $first
    skip: $skip
    orderBy: tradesQuantity
    orderDirection: desc
  ) {
    id
    tradesQuantity
    buysQuantity
    sellsQuantity
    collateralVolume
    scaledCollateralVolume
    collateralBuyVolume
    scaledCollateralBuyVolume
    collateralSellVolume
    scaledCollateralSellVolume
  }
}
""",
    default_variables={"first": 1000, "skip": 0},
)

MARKET_DATA = SubgraphQuery(
    name="marketDatas",
    subgraph="orderbook",
    description="Fetch MarketData entities mapping token IDs to conditions.",
    query="""\
query MarketDatas($first: Int!, $skip: Int!) {
  marketDatas(
    first: $first
    skip: $skip
    orderBy: id
    orderDirection: asc
  ) {
    id
    condition
    outcomeIndex
  }
}
""",
    default_variables={"first": 1000, "skip": 0},
)


# ===================================================================
# PnL subgraph queries
# ===================================================================

USER_POSITIONS = SubgraphQuery(
    name="userPositions",
    subgraph="pnl",
    description="Fetch user positions with average price and realized PnL.",
    query="""\
query UserPositions($first: Int!, $skip: Int!) {
  userPositions(
    first: $first
    skip: $skip
    orderBy: id
    orderDirection: asc
  ) {
    id
    user
    tokenId
    amount
    avgPrice
    realizedPnl
    totalBought
  }
}
""",
    default_variables={"first": 1000, "skip": 0},
)

PNL_CONDITIONS = SubgraphQuery(
    name="pnlConditions",
    subgraph="pnl",
    description="Fetch conditions with payout info from the PnL subgraph.",
    response_key="conditions",
    query="""\
query PnlConditions($first: Int!, $skip: Int!) {
  conditions(
    first: $first
    skip: $skip
    orderBy: id
    orderDirection: asc
  ) {
    id
    positionIds
    payoutNumerators
    payoutDenominator
  }
}
""",
    default_variables={"first": 1000, "skip": 0},
)

PNL_CONDITIONS_BY_ID = SubgraphQuery(
    name="pnlConditionsById",
    subgraph="pnl",
    description="Fetch conditions with payout info filtered by condition IDs.",
    response_key="conditions",
    query="""\
query PnlConditionsById($first: Int!, $skip: Int!, $conditionIds: [String!]) {
  conditions(
    first: $first
    skip: $skip
    orderBy: id
    orderDirection: asc
    where: { id_in: $conditionIds }
  ) {
    id
    positionIds
    payoutNumerators
    payoutDenominator
  }
}
""",
    default_variables={"first": 1000, "skip": 0, "conditionIds": []},
)


# ---------------------------------------------------------------------------
# Registry – look up queries by name
# ---------------------------------------------------------------------------

ALL_QUERIES: dict[str, SubgraphQuery] = {
    q.name: q
    for q in [
        MARKETS,
        CONDITIONS,
        POSITIONS,
        POSITIONS_BY_TOKEN_IDS,
        REDEMPTIONS,
        SPLITS,
        MERGES,
        NEG_RISK_EVENTS,
        NEG_RISK_CONVERSIONS,
        TRADES,
        TRADES_BY_MARKET,
        ORDER_FILLED_EVENTS,
        ORDERS_MATCHED_EVENTS,
        ORDERBOOK_STATS,
        MARKET_DATA,
        USER_POSITIONS,
        PNL_CONDITIONS,
        PNL_CONDITIONS_BY_ID,
    ]
}


def get_query(name: str) -> SubgraphQuery:
    """Retrieve a registered query by name, raising KeyError if unknown."""
    if name not in ALL_QUERIES:
        raise KeyError(
            f"Unknown query '{name}'. Available: {sorted(ALL_QUERIES.keys())}"
        )
    return ALL_QUERIES[name]
