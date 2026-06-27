\subsection{Data}

The LISA challenge dataset comprises 532 T2-weighted spin echo volumes from 244 pediatric subjects (age range: 0--24 months) acquired on a 0.064 T Hyperfine SWOOP portable MRI scanner. Imaging was conducted across three international sites in Uganda, South Africa, and the United States as part of the UNITY consortium, making this the first publicly available multi-site low-field brain MRI dataset. Each subject was scanned in three anisotropic orientations (axial, coronal, sagittal), producing three volumes per acquisition. A subset of subjects also had corresponding high-field (1.5 T or 3 T) T2-weighted scans registered to the low-field space, providing paired references for enhancement.

The dataset supports three complementary tasks, each leveraging distinct aspects of the data. The quality assessment task (Task 1a, \S\ref{sec:qa}) uses all 532 volumes with expert ratings across seven artifact classes to develop automated quality metrics for low-field MRI. The image enhancement task (Task 1b, \S\ref{sec:enhancement}) operates on the same low-field volumes and their high-field counterparts to learn a mapping from low- to high-quality image appearance. The segmentation task (Task 2, \S\ref{sec:seg}) provides expert-annotated labels for subcortical structures (hippocampus and basal ganglia) on a subset of the low-field scans, enabling automated volumetry in point-of-care settings. All volumes are provided as NIfTI files at their native resolution, with the three orientation-specific acquisitions treated as independent samples for QA and enhancement, and as multi-planar views for segmentation.

\subsection{Radiomics-based ULF-MRI Quality Assessment}
\label{sec:qa}

The quality assurance subset was made up of 532 T2-weighted volumes from 244 unique subjects, each acquired in three anisotropic orientations (axial, coronal, sagittal). Every volume was labeled with seven artifact categories (noise, zipper, positioning, banding, motion, contrast, and distortion), each rated on a three-point ordinal scale (0: no artifact, 1: minor artifact, 2: severe artifact). This formulation yields a multi-label, multi-class problem with 21 binary subproblems. All 532 volumes were used for model training to maximize the limited annotated data, with no held-out calibration split.

A radiomics\cite{pyrad} approach was adopted for quality assurance, leveraging handcrafted whole-image features to capture texture and intensity characteristics associated with each artifact type. Ninety-three features were extracted per volume, comprising first-order statistics (18), gray-level co-occurrence matrix (GLCM, 24), gray-level run-length matrix (GLRLM, 16), gray-level size-zone matrix (GLSZM, 16), gray-level dependence matrix (GLDM, 14), and neighboring gray-tone difference matrix (NGTDM, 5) features. A foreground mask was generated automatically using an intensity threshold to constrain feature extraction to the brain parenchyma. Features were extracted independently from each of the three orthogonal orientations and separate models were trained per view, enabling each orientation's unique artifact presentation to inform orthogonal predictions before ensembling.

Per-view feature selection was applied using variance thresholding followed by correlation-based filtering (Pearson r $>$ 0.95), removing near-zero-variance and highly redundant features. This reduced the initial 93 features to approximately 66 per view, retaining the most discriminative radiomic signatures. Classification was performed using XGBoost with one ensemble per label ($N=7$) per view ($N=3$), resulting in 21 models in total. Each subproblem received an individual hyperparameter search via cross-validation ($k=2$, 10 iterations), with class weights applied to account for the severe ordinal imbalance where class 0 dominates the majority of artifact categories. During inference, the per-view probability estimates were averaged across the three orientations and argmax assigned the final ordinal class.

\subsection{QA Results}
\label{sec:qa_results}

The radiomics-based QC model was evaluated on the challenge validation set (114 unlabeled volumes) through the LISA challenge platform. The initial submission achieved an F1-micro of 0.798 and an F1-macro of 0.463 (Table~\ref{tab:qa_results}). The substantial gap between micro and macro performance reflects the severe ordinal class imbalance: macro averaging weights each class equally, exposing that the model performs well on the dominant class 0 but struggles on the sparser classes 1 and 2 across all seven artifact types.

\begin{table}[h]
\centering
\caption{Quality assessment performance on the LISA challenge validation set.}
\label{tab:qa_results}
\begin{tabular}{lcc}
\toprule
\textbf{Model Variant} & \textbf{F1-micro} & \textbf{F1-macro} \\
\midrule
Radiomics XGBoost (initial, with calibration) & 0.798 & 0.463 \\
Radiomics XGBoost (no calibration, full data) & ---$^\dagger$ & ---$^\dagger$ \\
\bottomrule
\end{tabular}
\par\small\itshape
$^\dagger$Not yet evaluated on challenge platform; prediction divergence is 9.3\% relative to the calibrated variant, with Zipper class-2 predictions rising from 0 to 2 and Distortion class-1 tripling from 1 to 3.
\end{table}

The initial pipeline included Platt scaling on a held-out calibration split (15\% of the training set), which was found to collapse probability distributions and reduce minority-class recall. Removing the calibration step and folding those samples back into training produced a 9.3\% prediction change relative to the calibrated model, with notable increases in class 1 and class 2 predictions for Zipper and Distortion. This suggests the calibration holdout was too small to reliably fit a scaling function for highly imbalanced ordinal classes, and the additional training samples were more impactful than calibrated probabilities in this data-limited setting.

\paragraph{Feature importance analysis.}
Radiomic families are retained at markedly different rates during per-view feature selection (Figure~\ref{fig:selection_rate}). NGTDM features are the most compact and universally retained (15/15, 100\%), reflecting the high information density of its five texture measures. First-order statistics follow closely (42/54, 78\%), while GLCM features are most aggressively pruned (42/72, 58\%), suggesting substantial redundancy across its 24 descriptors at the whole-volume level.

\begin{figure}[h]
\centering
\includegraphics[width=0.65\linewidth]{figures/feature_selection_rate.pdf}
\caption{Feature survival rate per radiomic family after variance thresholding and correlation-based selection (Pearson $r > 0.95$), pooled across all three views. NGTDM has the highest retention (100\%) and GLCM the lowest (58\%).}
\label{fig:selection_rate}
\end{figure}

When ranking features by their mean XGBoost importance across views, GLSZM and first-order features jointly dominate, filling 20 of 35 top-5 slots (57\% combined). GLSZM matches first-order despite having two fewer available features (16 vs. 18), indicating that size-zone texture is disproportionately informative for ULF-MRI QC. GLCM claims 7 slots (20\%), GLDM 4 (11\%), NGTDM 3 (9\%), and GLRLM 1 (3\%).

\begin{table}[h]
\centering
\caption{Top-5 radiomic features per artifact class with mean importance scores.}
\label{tab:top_features}
\begin{tabular}{llcr}
\toprule
\textbf{Artifact} & \textbf{Feature} & \textbf{Family} & \textbf{Importance} \\
\midrule
\multirow{5}{*}{Noise}
 & Coarseness                                   & NGTDM   & 0.0485 \\
 & SmallAreaLowGrayLevelEmphasis                 & GLSZM   & 0.0311 \\
 & JointEnergy                                   & GLCM    & 0.0306 \\
 & Kurtosis                                      & First-order & 0.0269 \\
 & Busyness                                      & NGTDM   & 0.0259 \\
\midrule
\multirow{5}{*}{Zipper}
 & Energy                                        & First-order & 0.0321 \\
 & RobustMeanAbsoluteDeviation                   & First-order & 0.0281 \\
 & Idn                                           & GLCM    & 0.0235 \\
 & ZoneEntropy                                   & GLSZM   & 0.0230 \\
 & SizeZoneNonUniformityNormalized               & GLSZM   & 0.0222 \\
\midrule
\multirow{5}{*}{Positioning}
 & Skewness                                      & First-order & 0.0684 \\
 & Median                                        & First-order & 0.0323 \\
 & Mean                                          & First-order & 0.0294 \\
 & RobustMeanAbsoluteDeviation                   & First-order & 0.0251 \\
 & LargeAreaHighGrayLevelEmphasis                & GLSZM   & 0.0234 \\
\midrule
\multirow{5}{*}{Banding}
 & GrayLevelNonUniformityNormalized              & GLRLM   & 0.0558 \\
 & ClusterProminence                             & GLCM    & 0.0503 \\
 & Median                                        & First-order & 0.0416 \\
 & DependenceNonUniformityNormalized             & GLDM    & 0.0413 \\
 & Entropy                                       & First-order & 0.0383 \\
\midrule
\multirow{5}{*}{Motion}
 & SizeZoneNonUniformity                         & GLSZM   & 0.0441 \\
 & Correlation                                   & GLCM    & 0.0341 \\
 & HighGrayLevelZoneEmphasis                     & GLSZM   & 0.0233 \\
 & Contrast                                      & GLCM    & 0.0221 \\
 & ngtdm\_Contrast                               & NGTDM   & 0.0203 \\
\midrule
\multirow{5}{*}{Contrast}
 & SizeZoneNonUniformity                         & GLSZM   & 0.0499 \\
 & Correlation                                   & GLCM    & 0.0466 \\
 & SmallDependenceHighGrayLevelEmphasis          & GLDM    & 0.0428 \\
 & SizeZoneNonUniformityNormalized               & GLSZM   & 0.0333 \\
 & ZoneEntropy                                   & GLSZM   & 0.0307 \\
\midrule
\multirow{5}{*}{Distortion}
 & Contrast                                      & GLCM    & 0.0467 \\
 & LargeAreaEmphasis                             & GLSZM   & 0.0323 \\
 & RobustMeanAbsoluteDeviation                   & First-order & 0.0306 \\
 & SmallDependenceEmphasis                       & GLDM    & 0.0297 \\
 & SmallDependenceHighGrayLevelEmphasis          & GLDM    & 0.0247 \\
\bottomrule
\end{tabular}
\end{table}

Several features appear consistently across artifact types. \textit{SizeZoneNonUniformity} (GLSZM) and \textit{Correlation} (GLCM) are among the top discriminators for both Motion and Contrast artifacts, reflecting how tissue texture heterogeneity and voxel-wise intensity correlation degrade under these distortion types. \textit{Coarseness} (NGTDM) is the single most important feature for Noise detection (importance 0.0485, 1.6$\times$ above the next feature), capturing the fine-grained spatial intensity variation characteristic of high-frequency noise. First-order statistics---particularly \textit{Skewness} (0.0684, the highest importance across all artifact-feature pairs), \textit{Median}, and \textit{Mean}---dominate Positioning and, to a lesser extent, Zipper, consistent with artifacts that shift the global intensity distribution rather than locally alter texture.

Of the 21 subproblems, Positioning (axial variance) and Noise (coronal variance) exhibited the highest within-view feature variance, suggesting these artifacts manifest most heterogeneously across orientations, while Banding and Contrast were the most view-consistent.

Beyond the radiomics baseline, we explored three encoder-based QC architectures (\textit{Primus}, \textit{Conv3D}, and \textit{ReconFeature}) that learn features directly from volumes. These models are not yet evaluated due to GPU requirements, but the ReconFeature variant---freezing a 72M-parameter reconstruction-pretrained convolutional stem while training only a lightweight 300K-parameter head---offers a promising middle ground between handcrafted features and fully end-to-end learning that may capture artifact patterns invisible to fixed radiomic descriptors.

\subsection{ULF-MRI Quality Improvement}
\label{sec:enhancement}

\subsection{Neuranatomy Segmentation in ULF-MRI}
\label{sec:seg}
