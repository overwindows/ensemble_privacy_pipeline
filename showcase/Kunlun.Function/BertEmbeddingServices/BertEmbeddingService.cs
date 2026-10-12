// Copyright (c) Microsoft. All rights reserved.

using System.Diagnostics;
using System.Text;
using System.Text.Json;
using GSP.GenevaTelemetry.Counters.Tags;
using Kunlun.Function;
using Microsoft.Extensions.Options;

namespace Kunlun.Services;

/// <summary>
/// Self-contained DLIS BERT embedding client with mTLS certificate authentication,
/// round-robin endpoint selection, and Polly retry.
/// </summary>
internal sealed class BertEmbeddingService : IBertEmbeddingService
{
    internal const string HttpClientName = "BertEmbedding";

    private readonly IHttpClientFactory httpClientFactory;
    private readonly BertEmbeddingOptions options;
    private readonly IReadOnlyList<string> endpoints;
    private readonly ILogger<BertEmbeddingService> logger;
    private readonly string httpClientName;
    private int endpointIndex = -1;

    public BertEmbeddingService(
        IHttpClientFactory httpClientFactory,
        IOptions<BertEmbeddingOptions> options,
        ILogger<BertEmbeddingService> logger)
        : this(httpClientFactory, options, logger, HttpClientName)
    {
    }

    internal BertEmbeddingService(
        IHttpClientFactory httpClientFactory,
        IOptions<BertEmbeddingOptions> options,
        ILogger<BertEmbeddingService> logger,
        string httpClientName)
    {
        ArgumentNullException.ThrowIfNull(httpClientFactory, nameof(httpClientFactory));
        ArgumentNullException.ThrowIfNull(options?.Value, nameof(options));
        ArgumentNullException.ThrowIfNull(logger, nameof(logger));

        this.httpClientFactory = httpClientFactory;
        this.options = options.Value;
        endpoints = this.options.GetAllEndpoints();
        this.logger = logger;
        this.httpClientName = httpClientName;
    }

    /// <inheritdoc/>
    public async Task<List<float>> GetEmbeddingAsync(string title, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(title, nameof(title));

        if (title.Length > options.MaxTitleLength)
        {
            throw new ArgumentException(
                $"Title length ({title.Length}) exceeds maximum allowed ({options.MaxTitleLength}).",
                nameof(title));
        }

        var endpoint = GetNextEndpoint();
        var methodScope = new MethodScopeParameters("BertEmbedding.GetEmbedding", endpoint);
        using var scope = logger.BeginScope(methodScope);
        var stopwatch = Stopwatch.StartNew();

        try
        {
            var httpClient = httpClientFactory.CreateClient(httpClientName);
            httpClient.Timeout = TimeSpan.FromMilliseconds(options.HttpConnectionTimeoutInMs);

            var jsonContent = JsonSerializer.Serialize(new Dictionary<string, string>
            {
                ["title"] = title,
            });

            using var requestContent = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            using var response = await httpClient.PostAsync(endpoint, requestContent, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            var embedding = ParseEmbeddingResponse(responseJson);

            stopwatch.Stop();
            logger.LogBertEmbeddingSuccess(
                endpoint,
                stopwatch.ElapsedMilliseconds,
                embedding.Count);
            methodScope.UpdateMethodStatusAndReason(ExecutionStatus.Succeeded);

            return embedding;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            logger.LogError(
                ex,
                "BertEmbedding: failed. Endpoint={Endpoint}, LatencyMs={LatencyMs}",
                endpoint,
                stopwatch.ElapsedMilliseconds);
            methodScope.UpdateMethodStatusAndReason(ExecutionStatus.Failed, ex.GetType().Name);
            throw;
        }
    }

    private string GetNextEndpoint()
    {
        // Thread-safe round-robin: mask with int.MaxValue to handle overflow safely.
        var index = Interlocked.Increment(ref endpointIndex) & int.MaxValue;
        return endpoints[index % endpoints.Count];
    }

    private static List<float> ParseEmbeddingResponse(string responseJson)
    {
        Dictionary<string, List<float>>? dict;
        try
        {
            dict = JsonSerializer.Deserialize<Dictionary<string, List<float>>>(responseJson);
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException(
                "BERT DLIS response is not valid JSON or does not match expected schema.", ex);
        }

        if (dict is null || !dict.TryGetValue("low_dim_emb", out var embedding))
        {
            throw new InvalidOperationException(
                "BERT DLIS response does not contain the expected 'low_dim_emb' key.");
        }

        if (embedding.Count == 0)
        {
            throw new InvalidOperationException(
                "BERT DLIS response contains an empty 'low_dim_emb' vector.");
        }

        return embedding;
    }
}