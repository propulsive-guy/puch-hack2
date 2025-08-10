# mcp_server.py
import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---


TOKEN = "synapse07"
MY_NUMBER = "917499968162"
GEMINI_API_KEY = "AIzaSyDN435fBaO-R8GtfD68wpWrGwsAxxU6j40"

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class (kept from your code, with minor tweaks) ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of result URLs.
        (Using DuckDuckGo HTML frontend to avoid scraping blocks.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        # DuckDuckGo 'result__a' anchors typically contain the link
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (kept) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# --- Tool: make_img_black_and_white (kept) ---
MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
    

FIND_PRODUCT_ONLINE_DESCRIPTION = "Identify a product from an uploaded WhatsApp image and search online shopping sites for it."

@mcp.tool(description=FIND_PRODUCT_ONLINE_DESCRIPTION)
async def find_product_online(
    caption: Annotated[str, Field(description="Caption text from WhatsApp (starts with tool name, then search query)")],
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data from WhatsApp to identify product")] = None,
    max_search_results: Annotated[int, Field(description="Max number of search results to fetch", default=10)] = 10,
) -> str:
    """
    Identify the product in a WhatsApp-uploaded image and return shopping links.
    Falls back to text-only search if the image can't be processed.
    """
    import base64
    import os
    import httpx
    import json
    from typing import List

    # --- Extract user query from caption ---
    user_prompt = caption.strip()
    for prefix in (
        "find_product_online", "find_product_by_image",
        "user_tool_find_product_online", "user_tool_find_product_by_image"
    ):
        if user_prompt.lower().startswith(prefix):
            user_prompt = user_prompt[len(prefix):].strip()
            break

    product_description = None

    try:
        if puch_image_data:
            # Clean and validate base64 image data
            image_data_clean = puch_image_data.strip()
            # Remove data URL prefix if present
            if image_data_clean.startswith('data:image/'):
                image_data_clean = image_data_clean.split(',', 1)[1]
            
            # Decode & validate image
            try:
                image_bytes = base64.b64decode(image_data_clean)
                print(f"[INFO] Successfully decoded image: {len(image_bytes)} bytes")
            except Exception as decode_error:
                print(f"[ERROR] Failed to decode base64 image: {decode_error}")
                raise decode_error

            # Re-encode for API
            product_image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Get API key from environment
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                raise Exception("Missing GEMINI_API_KEY in environment variables")

            # Detect image format (simple detection)
            mime_type = "image/jpeg"  # default
            if image_bytes[:4] == b'\x89PNG':
                mime_type = "image/png"
            elif image_bytes[:3] == b'\xFF\xD8\xFF':
                mime_type = "image/jpeg"
            elif image_bytes[:6] == b'GIF87a' or image_bytes[:6] == b'GIF89a':
                mime_type = "image/gif"
            elif image_bytes[:4] == b'WEBP':
                mime_type = "image/webp"

            print(f"[INFO] Detected image type: {mime_type}")

            # Enhanced instruction for better product identification
            instruction_text = (
               ''' You are a product-identification expert. Analyze the image carefully and return ONE concise shopping search query (max 10 words) that will most likely find the exact item online.
               extract shape , logos, text, and any visible features and colour to identify the exact product in the image

Rules:
1. Produce a short, search-friendly phrase: include category + the most specific visible identifiers (brand or model if visible), dominant color, material or flavor, size/volume/weight/count when present, and any distinctive feature (pattern, packaging text, gender cut).
2. If label/packaging text is visible, include the exact words (put nothing else — transcription only).
3. Do NOT invent brands, model numbers, or attributes the image does not show. Prefer visible text; otherwise pick the most discriminative visual attributes.
4. For food include flavor and package size when visible (e.g., "mango yogurt 150g"). For electronics include model number if visible (e.g., "sony wh-1000xm5 headphones").
5. Use lowercase, no punctuation except hyphens, and keep it compact (≤10 words).
6. If multiple items are present, describe the primary/center item only.
Return ONLY the final single-line query and nothing else'''

            )

            # Gemini 2.5 Pro API URL
            gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{instruction_text}\n\nUser's additional context: {user_prompt}" if user_prompt else instruction_text},
                            {
                                "inline_data": {  # Correct field name for newer API
                                    "mime_type": mime_type,
                                    "data": product_image_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "candidateCount": 1,
                    "maxOutputTokens": 64,
                    "topP": 0.8
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT", 
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"  
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            }

            print(f"[INFO] Calling Gemini API for image analysis...")

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(gemini_url, headers=headers, json=payload)

            print(f"[INFO] Gemini API response status: {resp.status_code}")

            if resp.status_code == 200:
                gemini_json = resp.json()
                print(f"[DEBUG] Gemini response: {json.dumps(gemini_json, indent=2)}")
                
                candidates = gemini_json.get("candidates", [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts and len(parts) > 0:
                        text_part = parts[0].get("text", "").strip()
                        if text_part:
                            # Clean the response - take first line only
                            product_description = text_part.splitlines()[0].strip()
                            print(f"[SUCCESS] Product identified: {product_description}")
                        else:
                            print("[WARN] Empty text response from Gemini")
                    else:
                        print("[WARN] No parts in Gemini response")
                else:
                    print("[WARN] No candidates in Gemini response")
            else:
                error_text = resp.text
                print(f"[ERROR] Gemini API error {resp.status_code}: {error_text}")
                # Try to parse error for more details
                try:
                    error_json = resp.json()
                    print(f"[ERROR] Detailed error: {json.dumps(error_json, indent=2)}")
                except:
                    pass

    except Exception as e:
        print(f"[WARN] Image processing failed, falling back to text-only search. Error: {str(e)}")
        import traceback
        traceback.print_exc()

    # Fallback to text search if image processing failed
    if not product_description:
        if user_prompt:
            product_description = user_prompt
            print(f"[INFO] Using text fallback: {product_description}")
        else:
            product_description = "product"
            print("[INFO] No description available, using generic 'product'")

    # Enhanced search query construction
    search_terms = []
    if "buy" not in product_description.lower():
        search_terms.append("buy")
    search_terms.append(product_description)
    search_terms.append("online")
    
    search_query = " ".join(search_terms).strip()[:240]
    print(f"[INFO] Final search query: {search_query}")

    # Perform search
    try:
        links = await Fetch.google_search_links(search_query, num_results=max_search_results)
        print(f"[INFO] Found {len(links)} initial search results")
    except Exception as search_error:
        print(f"[ERROR] Search failed: {search_error}")
        return f"❌ **Error:** Failed to search for products. Please try again.\n\n**Product identified:** {product_description}"

    # Enhanced shopping domain list (India-focused + international)
    shopping_domains = [
        # Indian platforms
        "amazon.in", "flipkart.com", "myntra.com", "ajio.com", "snapdeal.com", 
        "meesho.com", "paytmmall.com", "nykaa.com", "firstcry.com", "bigbasket.com",
        "grofers.com", "shopclues.com", "jabong.com", "koovs.com", "limeroad.com","zomato.com"
        
        # International platforms  
        "amazon.com", "ebay.com", "alibaba.com", "aliexpress.com", "lazada.com",
        "shopee.com", "asos.com", "zara.com", "hm.com", "uniqlo.com", "nike.com",
        "adidas.com", "lightinthebox.com", "banggood.com", "gearbest.com"
    ]
    
    # Categorize links
    shopping_links = []
    other_links = []
    
    for link in links:
        link_lower = link.lower()
        is_shopping = any(domain in link_lower for domain in shopping_domains)
        
        if is_shopping:
            shopping_links.append(link)
        else:
            other_links.append(link)

    print(f"[INFO] Categorized results - Shopping: {len(shopping_links)}, Other: {len(other_links)}")

    # If no shopping links found, try alternative searches
    if not shopping_links:
        print("[INFO] No shopping links found, trying alternative searches...")
        alternative_queries = [
            f"{product_description} price",
            f"{product_description} shop online",
            f"where to buy {product_description}",
            f"{product_description} flipkart amazon"
        ]
        
        for alt_query in alternative_queries:
            try:
                alt_links = await Fetch.google_search_links(alt_query, num_results=5)
                for link in alt_links:
                    link_lower = link.lower()
                    if any(domain in link_lower for domain in shopping_domains):
                        if link not in shopping_links:
                            shopping_links.append(link)
                    elif link not in other_links:
                        other_links.append(link)
                
                if shopping_links:  # Break if we found some shopping links
                    break
            except Exception as alt_search_error:
                print(f"[WARN] Alternative search failed: {alt_search_error}")
                continue

    # Format response
    header = f"🛒 **Product identified:** {product_description}\n"
    header += f"🔍 **Search query:** `{search_query}`\n\n"

    if shopping_links:
        body = "🛍️ **Shopping Links:**\n"
        for i, link in enumerate(shopping_links[:8], 1):  # Limit to top 8
            body += f"{i}. {link}\n"
        
        if other_links:
            body += "\n🔎 **Additional Results:**\n"
            for i, link in enumerate(other_links[:3], 1):  # Show top 3 other results
                body += f"{i}. {link}\n"
    else:
        body = "⚠️ **No direct shopping links found**\n\n"
        body += "💡 **Suggestions:**\n"
        body += "• Try a more specific caption (e.g., 'red cotton t-shirt men large')\n"
        body += "• Upload a clearer image with better lighting\n"
        body += "• Include brand name if visible\n\n"
        
        if other_links:
            body += "🔍 **Search Results:**\n"
            for i, link in enumerate(other_links[:5], 1):
                body += f"{i}. {link}\n"

    final_response = header + body
    print(f"[INFO] Response generated, length: {len(final_response)} characters")
    return final_response


# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
