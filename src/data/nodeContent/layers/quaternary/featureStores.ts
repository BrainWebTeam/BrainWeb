import { NodeContent } from '../../../../types/content';

export const featureStoresContent: NodeContent = {
  title: 'Feature Stores',
  description: 'A centralized platform for storing, managing, and serving machine learning features, ensuring consistency between training and inference.',
  concepts: [
    'Feature Management',
    'Online/Offline Storage',
    'Feature Versioning',
    'Point-in-time Correctness',
    'Feature Sharing'
  ],
  examples: [
    {
      language: 'python',
      description: 'Feature store implementation',
      code: `from feast import FeatureStore, Entity, Feature, ValueType
from feast.feature_view import FeatureView
from feast.data_source import FileSource

# Define an entity
driver = Entity(
    name="driver",
    value_type=ValueType.INT64,
    description="Driver ID"
)

# Define a feature view
driver_stats_fv = FeatureView(
    name="driver_stats",
    entities=["driver"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="trips_today", dtype=ValueType.INT64),
        Feature(name="avg_rating", dtype=ValueType.FLOAT)
    ],
    online=True,
    input=FileSource(
        path="data/driver_stats.parquet",
        event_timestamp_column="event_timestamp"
    )
)

# Initialize and apply
store = FeatureStore("feature_repo/")
store.apply([driver, driver_stats_fv])`
    }
  ],
  resources: [
    {
      title: 'Feature Store Guide',
      description: 'Introduction to ML feature stores',
      url: 'https://www.feast.dev/blog/what-is-a-feature-store/'
    },
    {
      title: 'Feature Store Design',
      description: 'Designing feature stores for ML',
      url: 'https://www.tecton.ai/blog/feature-store-design/'
    }
  ],
  prerequisites: ['Data Engineering', 'MLOps', 'Data Management'],
  relatedTopics: ['Data Pipeline', 'Feature Engineering', 'ML Infrastructure']
};