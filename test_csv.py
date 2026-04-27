import pandas as pd
df_fire = pd.read_csv("firestation_sample.csv")
df_fire['longitude'] = df_fire['geometry'].str.extract(r'POINT \(([-\d.]+)', expand=False)
df_fire['latitude'] = df_fire['geometry'].str.extract(r'POINT \([-\d.]+ ([-\d.]+)\)', expand=False)
print(df_fire[['geometry', 'longitude', 'latitude', 'name']])
