using System.Text.Json;
using System.Text.Json.Serialization;
using GSP.DlisInference;

namespace Kunlun.AgenticGem.Services;

internal sealed class InterestReranker(DlisInterestRerankerModelService reranker)
{
    private const double DecisionThreshold = 0.5;

    private const string PromptTemplate = """
<Instruct>: You are evaluating two user interest nodes in a taxonomy hierarchy (`Category` > `Persona`). Each interest node describes:
- **Category**: The broad topic or subject area (e.g., "AI Assistant Products", "Travel and Hospitality", "Automotive Enthusiasts")
- **Persona**: A concise user intent or behavioral description (e.g., "Using AI for productivity", "Planning travel and dining", "Following automotive news")

Note that Interest 1 and Interest 2 MAY NOT share the same Category.

Your task is to determine whether Interest 1 and Interest 2 are "near-duplicates" that should be merged into the exact same cluster.

CRITICAL - OUTPUT FORMAT: Output ONLY "yes" or "no". Do NOT include any reasoning, punctuation, spaces, or extra characters.

CRITICAL - ORDER INDEPENDENCE: Your judgment MUST be perfectly symmetric. Swapping the order of Interest 1 and Interest 2 must produce the exact same outcome.

---

### Core Deduplication Rules:

1. **MERGE ("yes") - Strict Semantic Equivalence & Same-Entity Intent Overlap**:
   - **Synonym & Paraphrase**: Merge if categories and personas are phrasing or organizational variations of each other.
   - **Cross-Category Equivalence**: Merge if personas describe identical behavioral intent under semantically equivalent categories.
   - **Same Specific Entity + Overlapping Intent**: If BOTH nodes target the EXACT SAME specific entity, brand, location, or company (e.g., both target "SpaceX" or both target "Japan"), AND their behavioral intents strongly overlap (e.g., "monitoring financial valuation" vs. "tracking corporate and investor news"), they MUST BE MERGED (`yes`).

2. **DO NOT MERGE ("no") - Entity / Brand / Location Mismatch (CRITICAL)**:
   - **Different Brands/Companies/Products**: Do NOT merge different branded products, companies, or services, even if in the same industry (e.g., "SpaceX" vs. "Blue Origin" -> `no`; "Google Gemini" vs. "Microsoft Copilot" -> `no`).
   - **Different Locations**: Do NOT merge interests focusing on different specific locations (e.g., "Japan Travel" vs. "Beijing Travel" -> `no`).
   - **Different Individuals/Teams**: Do NOT merge distinct individual entities (e.g., "Lionel Messi" vs. "Cristiano Ronaldo" -> `no`).

3. **DO NOT MERGE ("no") - Granularity & Scope Mismatch**:
   - **Niche vs. Generic Broad**: Do NOT merge a broad/generic industry category with a specific entity-restricted interest (e.g., "General Aerospace Industry" vs. "SpaceX Corporate News" -> `no`; "General Travel Planning" vs. "Japan Travel Planning" -> `no`).
   - **Sibling Concepts (Co-hyponyms)**: Do NOT merge two different sub-topics under the same parent category (e.g., "Python Programming" vs. "Java Programming" -> `no`).

4. **DO NOT MERGE ("no") - Intent & Role Conflicts**:
   - Do NOT merge differing functional intents or roles within the same topic (e.g., "Consumer/Enthusiast" vs. "Service Provider/B2B", "Buying" vs. "Renting").

---

### Key Decision Rule:
- **Same Specific Entity (e.g., both SpaceX, both Japan) + Overlapping Intent?** -> **yes**
- **Different Specific Entities (e.g., SpaceX vs. Blue Origin, Japan vs. Beijing, Gemini vs. Copilot)?** -> **no**
- **Generic Industry/Category vs. Specific Entity (e.g., General Aerospace vs. SpaceX)?** -> **no**

---

### Examples:

Example 1:
Interest 1: SpaceX Corporate News > Monitoring SpaceX's financial valuation and market position.
Interest 2: SpaceX > Tracking SpaceX corporate and investor news.
Judgment: yes

Example 2:
Interest 1: Japan Travel Planning > Planning an upcoming trip to Japan.
Interest 2: Japan Travel and Culture > Exploring or planning travel and life in Japan.
Judgment: yes

Example 3:
Interest 1: ChatGPT and OpenAI AI Tools > Using ChatGPT for various tasks.
Interest 2: OpenAI ChatGPT Products > Leveraging ChatGPT for task automation.
Judgment: yes

Example 4:
Interest 1: Google Gemini AI > Leveraging Google Gemini for productivity and organization.
Interest 2: Microsoft Copilot > Exploring Microsoft Copilot AI capabilities.
Judgment: no

Example 5:
Interest 1: Japan Travel Planning > Planning an upcoming trip to Japan.
Interest 2: Beijing Travel Planning > Planning a trip to Beijing.
Judgment: no

Example 6:
Interest 1: Commercial Spaceflight Industry > Following news about space exploration and aerospace companies.
Interest 2: SpaceX > Tracking SpaceX corporate and investor news.
Judgment: no

---

Now evaluate the following two interest nodes and output ONLY "yes" or "no":
Interest 1: {interest1}
Interest 2: {interest2}
Judgment:
""";

    public async Task<IReadOnlyList<InterestRerankerResult>> GetResultsAsync(
        string currentInterest,
        IReadOnlyList<string> candidateInterests,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (candidateInterests.Count == 0)
        {
            return [];
        }

        var request = new InterestRerankerRequest(
            candidateInterests.Select(candidate => new InterestPair(currentInterest, candidate)).ToList(),
            DecisionThreshold,
            PromptTemplate);
        var responseJson = await reranker.GetResponseAsync(JsonSerializer.Serialize(request)).ConfigureAwait(false);
        var response = JsonSerializer.Deserialize<InterestRerankerResponse>(responseJson);
        if (response?.Results is null)
        {
            throw new FormatException($"Unexpected reranker response: '{responseJson}'.");
        }

        if (response.Results.Count != candidateInterests.Count)
        {
            throw new FormatException(
                $"Expected {candidateInterests.Count} reranker results, but received {response.Results.Count}.");
        }

        if (response.Results.Any(result => !double.IsFinite(result.Score) || result.Score is < 0.0 or > 1.0))
        {
            throw new FormatException($"Unexpected reranker score in response: '{responseJson}'.");
        }

        return response.Results;
    }

    private sealed record InterestRerankerRequest(
        [property: JsonPropertyName("pairs")] IReadOnlyList<InterestPair> Pairs,
        [property: JsonPropertyName("threshold")] double Threshold,
        [property: JsonPropertyName("prompt_template")] string PromptTemplate);

    private sealed record InterestPair(
        [property: JsonPropertyName("interest1")] string Interest1,
        [property: JsonPropertyName("interest2")] string Interest2);

    private sealed record InterestRerankerResponse(
        [property: JsonPropertyName("results")] IReadOnlyList<InterestRerankerResult> Results);
}