using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using GSP.DlisInference;

using Microsoft.Extensions.Logging;

namespace GSP.Gateway.Services.GemGrounding;

internal static class GemReferDocRerankUtils
{
    private const string SystemPrompt = "<|im_start|>system\\nJudge whether the Document meets the requirements based on the Query.\r\nRequirements:\r\n1. Relevance – The document must be highly relevant to the query.\r\n2. Source Quality – Prefer documents from well-recognized publishers with strong reputations.\r\n3. Domain Fit – Prefer publishers that specialize in the event’s topic area. \r\nNote that the answer can only be \\\"yes\\\" or \\\"no\\\".<|im_end|>\\n<|im_start|>user";
    private const string Instruction = "Given a web search query, retrieve relevant passages that answer the query. Besides relevance, you should also prefer passage from good publisher";
    private const int GroupSize = 5;

    public static async Task<List<double>> ProcessBatchInferenceAsync(
        ILogger logger,
        DlisGemReferDocRerankModelService gemDocRerankModelService,
        DlisReferDocRerankModelService2 gemDocRerankModelService2,
        List<GemReferDocRerankScoreBatchRequest> batch)
    {
        if (batch is not { Count: > 0 })
            return [];

        var stopwatch = Stopwatch.StartNew();
        try
        {
            var numGroups = (batch.Count + GroupSize - 1) / GroupSize;
            var baseCount = batch.Count / numGroups;
            var remainder = batch.Count % numGroups;

            var groupTasks = new Task<List<double>>[numGroups];
            var cursor = 0;
            for (var i = 0; i < numGroups; i++)
            {
                var size = baseCount + (i < remainder ? 1 : 0);
                var groupItems = batch.GetRange(cursor, size);
                cursor += size;
                groupTasks[i] = ProcessGroupAsync(logger, gemDocRerankModelService, gemDocRerankModelService2, groupItems);
            }

            var groupResults = await Task.WhenAll(groupTasks).ConfigureAwait(false);
            var merged = new List<double>(batch.Count);
            foreach (var groupResult in groupResults)
                merged.AddRange(groupResult);

            stopwatch.Stop();
            logger.LogInformation(
                "DLIS Gem refer-doc rerank inference succeeded. elapsedMs={ElapsedMs}, batchSize={BatchSize}, groups={Groups}, scores={Scores}",
                stopwatch.ElapsedMilliseconds,
                batch.Count,
                numGroups,
                JsonSerializer.Serialize(merged));
            return merged;
        }
        catch (Exception exception)
        {
            stopwatch.Stop();
            logger.LogError(exception, "Failed to call DLIS rerank endpoint after {ElapsedMs}ms.", stopwatch.ElapsedMilliseconds);
            throw new InvalidOperationException(
                $"Failed to process reference doc rerank scoring after {stopwatch.ElapsedMilliseconds}ms",
                exception);
        }
    }

    private static async Task<List<double>> ProcessGroupAsync(
        ILogger logger,
        DlisGemReferDocRerankModelService gemDocRerankModelService,
        DlisReferDocRerankModelService2 gemDocRerankModelService2,
        List<GemReferDocRerankScoreBatchRequest> group)
    {
        var dataList = new List<object>(group.Count);
        foreach (var pair in group)
        {
            dataList.Add(new
            {
                query = pair.Query,
                chunk = pair.Text,
                system_prompt = SystemPrompt,
                instruction = Instruction
            });
        }

        var requestPayload = JsonSerializer.Serialize(dataList);
        logger.LogInformation("DLIS Gem refer-doc rerank request: {RequestPayload}", requestPayload);

        var response = Random.Shared.Next(3) < 1
            ? await gemDocRerankModelService.GetResponseAsync(requestPayload).ConfigureAwait(false)
            : await gemDocRerankModelService2.GetResponseAsync(requestPayload).ConfigureAwait(false);

        if (string.IsNullOrWhiteSpace(response))
            throw new InvalidOperationException("Empty response from rerank model.");

        var resultList = JsonSerializer.Deserialize<List<double>>(response);
        if (resultList is not { Count: > 0 } || resultList.Count != group.Count)
            throw new InvalidOperationException($"Rerank group returned {resultList?.Count ?? 0} scores, expected {group.Count}.");

        return resultList;
    }
}

internal sealed class GemReferDocRerankScoreBatchRequest
{
    [JsonPropertyName("query")]
    public required string Query { get; set; }

    [JsonPropertyName("text")]
    public required string Text { get; set; }
}

