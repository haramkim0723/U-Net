import os
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import io
from skimage.morphology import skeletonize

# 데이터 로드 (줄기 두께 데이터 직접 입력)
data = """
1	0.427924972
2	0.43321953
1	0.429347494
2	0.436686955
1	0.436976124
2	0.43928051
1	0.43835761
2	0.440059154
1	0.437363137
2	0.433460161
1	0.447700253
2	0.442037762
1	0.442673737
2	0.438133404
1	0.455854568
2	0.434341591
1	0.438174908
2	0.438274617
1	0.436405031
2	0.440442463
1	0.433666853
2	0.434059408
1	0.435047384
2	0.439112833
1	0.455172415
2	0.443876357
1	0.441927691
2	0.43717103
1	0.430468664
2	0.440806543
1	0.441388013
2	0.438689543
1	0.438982899
2	0.433195094
1	0.444812623
2	0.432442125
1	0.450730827
2	0.438945596
1	0.419615814
2	0.438322689
1	0.431606907
2	0.438733317
1	0.433890099
2	0.437827119
1	0.443732411
2	0.440900523
1	0.442207986
2	0.433916028
1	0.443128367
2	0.44092096
1	0.455383681
2	0.43953088
1	0.434039911
2	0.442413798
1	0.440141319
2	0.437821266
1	0.441689031
2	0.439008934
1	0.439086453
2	0.433406113
1	0.442627033
2	0.444391287
1	0.447220324
2	0.432125566
1	0.445150554
2	0.439475154
1	0.444485149
2	0.436547114
1	0.440408293
2	0.434510383
1	0.438055758
2	0.431552629
1	0.432525207
2	0.433659354
1	0.436174891
2	0.43942488
1	0.425869141
2	0.430306135
1	0.439835721
2	0.439592894
1	0.444339919
2	0.438629952
1	0.432797984
2	0.443122854
1	0.428329557
2	0.440126798
1	0.444123759
2	0.444616465
1	0.427187171
2	0.439174128
1	0.428797122
2	0.434445463
1	0.423336948
2	0.438379669
1	0.439909903
2	0.435977362
1	0.445567139
2	0.439180714
1	0.431864227
2	0.439901707
"""

# 데이터 읽기
data_io = io.StringIO(data.strip())
df = pd.read_csv(data_io, sep="\t", header=None, names=["set_index", "stem_real_cm"])

# 세트별 분리
df_set1 = df[df["set_index"] == 1].reset_index(drop=True)
df_set2 = df[df["set_index"] == 2].reset_index(drop=True)

df_split = pd.DataFrame({
    "trial": range(1, len(df_set1) + 1),
    "set1_stem_real_cm": df_set1["stem_real_cm"],
    "set2_stem_real_cm": df_set2["stem_real_cm"]
})

# 평균과 표준편차 계산
summary_stats = pd.DataFrame({
    "Set": ["Set 1", "Set 2"],
    "Mean (cm)": [df_split['set1_stem_real_cm'].mean(), df_split['set2_stem_real_cm'].mean()],
    "Standard Deviation (cm)": [df_split['set1_stem_real_cm'].std(), df_split['set2_stem_real_cm'].std()]
})

# 정규분포 곡선 그리기 준비
x_set1 = np.linspace(summary_stats.loc[0, "Mean (cm)"] - 3*summary_stats.loc[0, "Standard Deviation (cm)"],
                     summary_stats.loc[0, "Mean (cm)"] + 3*summary_stats.loc[0, "Standard Deviation (cm)"], 100)
y_set1 = stats.norm.pdf(x_set1, summary_stats.loc[0, "Mean (cm)"], summary_stats.loc[0, "Standard Deviation (cm)"])

x_set2 = np.linspace(summary_stats.loc[1, "Mean (cm)"] - 3*summary_stats.loc[1, "Standard Deviation (cm)"],
                     summary_stats.loc[1, "Mean (cm)"] + 3*summary_stats.loc[1, "Standard Deviation (cm)"], 100)
y_set2 = stats.norm.pdf(x_set2, summary_stats.loc[1, "Mean (cm)"], summary_stats.loc[1, "Standard Deviation (cm)"])

# 개별 그래프 시각화
plt.figure(figsize=(12, 5))

# Set 1 정규분포 그래프
plt.subplot(1, 2, 1)
plt.plot(x_set1, y_set1, label="Set 1", color='skyblue')
plt.fill_between(x_set1, y_set1, alpha=0.3, color='skyblue')
plt.title("Normal Distribution - Set 1")
plt.xlabel("Stem Thickness (cm)")
plt.ylabel("Probability Density")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Set 2 정규분포 그래프
plt.subplot(1, 2, 2)
plt.plot(x_set2, y_set2, label="Set 2", color='lightgreen')
plt.fill_between(x_set2, y_set2, alpha=0.3, color='lightgreen')
plt.title("Normal Distribution - Set 2")
plt.xlabel("Stem Thickness (cm)")
plt.ylabel("Probability Density")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.show()
