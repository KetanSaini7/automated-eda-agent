import pandas as pd

from src.anomaly import detect_anomalies


data = {
    "Sales": [
        100,
        120,
        110,
        130,
        115,
        125,
        140,
        135,
        150,
        5000
    ],

    "Profit": [
        10,
        12,
        11,
        13,
        15,
        14,
        16,
        17,
        18,
        200
    ]
}


df = pd.DataFrame(data)

print("Original Data:")
print(df)


# Run anomaly detection
results = detect_anomalies(df)


print("\n" + "=" * 50)
print("ANOMALY DETECTION RESULTS")
print("=" * 50)


print("\nIQR Results:")
for column, result in results["iqr"].items():

    print(f"\n{column}")
    print(f"Anomalies: {result['anomaly_count']}")
    print(
        f"Anomaly Percentage: "
        f"{result['anomaly_percentage']}%"
    )
    print(
        f"Lower Bound: "
        f"{result['lower_bound']}"
    )
    print(
        f"Upper Bound: "
        f"{result['upper_bound']}"
    )


print("\nMultivariate Isolation Forest Results:")
print("=" * 50)

result = results["multivariate_isolation_forest"]

if result["status"] == "skipped":

    print("Status: SKIPPED")
    print(f"Reason: {result['reason']}")

else:

    print("Status: COMPLETED")

    print(
        f"\nFeatures Used:"
    )

    print(
        result["features_used"]
    )

    print(
        "\nExcluded Features:"
    )

    for column, reason in result[
        "excluded_features"
    ].items():

        print(
            f"  {column} → {reason}"
        )

    print(
        f"\nRows Analyzed: "
        f"{result['rows_analyzed']:,}"
    )

    print(
        f"Anomalies: "
        f"{result['anomaly_count']:,}"
    )

    print(
        f"Anomaly Percentage: "
        f"{result['anomaly_percentage']}%"
    )

    print(
        f"\nFirst 10 Anomaly Indices:"
    )

    print(
        result["anomaly_indices"][:10]
    )



