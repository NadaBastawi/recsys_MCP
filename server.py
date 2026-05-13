"""
TORCO Pest Control — MCP Server
Exposes the recommendation engine as an MCP tool so AI agents
can query it in natural language.

An agent can ask:
  "What services should I recommend for customer C0001?"
  "What does a new residential customer in desert Tucson need most?"
  "Which customers have HIGH urgency this summer?"
"""

import json
import sys
import asyncio
import httpx
from typing import Any

# MCP server implementation using stdio transport
# Compatible with Claude Desktop, Continue, and any MCP client

TOOLS = [
    {
        "name": "recommend_for_customer",
        "description": (
            "Get personalized pest control service recommendations for an existing TORCO customer. "
            "Returns top services ranked by recommendation score and urgency. "
            "Use this when you know the customer ID."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer ID (e.g. C0001, C0042)"
                },
                "season": {
                    "type": "string",
                    "enum": ["spring", "summer", "fall", "winter"],
                    "description": "Current season — affects urgency scoring"
                },
                "days_since_last_service": {
                    "type": "integer",
                    "description": "Days since this customer's last service visit"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of recommendations to return (default 5)"
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "recommend_for_new_customer",
        "description": (
            "Get pest control service recommendations for a NEW customer with no history. "
            "Uses property profile for cold-start recommendations. "
            "Use this when onboarding a new customer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "property_type": {
                    "type": "string",
                    "enum": ["residential_house", "residential_apartment", "commercial_restaurant",
                             "commercial_office", "commercial_warehouse", "mobile_home"],
                    "description": "Type of property"
                },
                "zip_code": {
                    "type": "string",
                    "description": "Tucson ZIP code (e.g. 85743)"
                },
                "building_age_years": {
                    "type": "integer",
                    "description": "Age of the building in years"
                },
                "season": {
                    "type": "string",
                    "enum": ["spring", "summer", "fall", "winter"],
                    "description": "Current season"
                }
            },
            "required": ["property_type", "zip_code", "building_age_years"]
        }
    },
    {
        "name": "list_services",
        "description": "List all available TORCO pest control services with pricing.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]

API_BASE = "http://localhost:8000"

async def call_tool(name: str, arguments: dict) -> str:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            if name == "recommend_for_customer":
                resp = await client.post(f"{API_BASE}/recommend", json=arguments)
                data = resp.json()
                lines = [f"Recommendations for {data['customer_id']}:\n"]
                for i, r in enumerate(data["recommendations"], 1):
                    lines.append(
                        f"{i}. {r['service_name']} — ${r['price_usd']}\n"
                        f"   Score: {r['recommendation_score']} | "
                        f"Urgency: {r['urgency_score']}/10 [{r['urgency_label']}]"
                    )
                return "\n".join(lines)

            elif name == "recommend_for_new_customer":
                resp = await client.post(f"{API_BASE}/recommend/new-customer", json=arguments)
                data = resp.json()
                lines = [f"Cold-start recommendations (new customer):\n",
                         f"Profile: {data['profile']['property_type']} | ZIP {data['profile']['zip_code']}\n"]
                for i, r in enumerate(data["recommendations"], 1):
                    lines.append(
                        f"{i}. {r['service_name']} — ${r['price_usd']}\n"
                        f"   Urgency: {r['urgency_score']}/10 [{r['urgency_label']}]"
                    )
                return "\n".join(lines)

            elif name == "list_services":
                resp = await client.get(f"{API_BASE}/services")
                data = resp.json()
                lines = ["Available TORCO Services:\n"]
                for sid, info in data["services"].items():
                    lines.append(f"  {sid}: {info['service_name']} — ${info['price_usd']} ({info['category']})")
                return "\n".join(lines)

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            return f"Error calling API: {str(e)}. Make sure the FastAPI server is running on port 8000."


async def handle_message(message: dict) -> dict | None:
    method = message.get("method")
    msg_id = message.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "torco-recsys-mcp", "version": "1.0.0"}
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {"tools": TOOLS}
        }

    elif method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        result_text = await call_tool(tool_name, arguments)
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False
            }
        }

    elif method == "notifications/initialized":
        return None

    return {
        "jsonrpc": "2.0", "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    }


async def main():
    """Run MCP server over stdio"""
    while True:
        line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            response = await handle_message(message)
            if response is not None:
                print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32603, "message": str(e)}
            }), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
