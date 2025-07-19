

from datetime import timedelta
from feast import Entity, FeatureView, Field
from feast.types import Float32
from feast.infra.offline_stores.bigquery_source import BigQuerySource
from feast import ValueType

# ✅ Define the entity to match BigQuery column
iris_entity = Entity(
    name="entity_id",  # MUST match the BigQuery column
    join_keys=["entity_id"],
    value_type=ValueType.INT64,
    description="Iris sample identifier"
)

# ✅ Define your source
iris_source = BigQuerySource(
    table="iris_feature_store.iris_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# ✅ Define the FeatureView and reference the entity
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],  # ✅ This is where you include the entity
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="sepal_ratio", dtype=Float32),
        Field(name="petal_ratio", dtype=Float32),
        Field(name="total_area", dtype=Float32),
    ],
    online=True,
    source=iris_source,
    tags={"team": "ml_engineering"},
)