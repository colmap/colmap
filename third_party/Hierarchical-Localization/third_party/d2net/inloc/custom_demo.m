% Startup
startup;
[ params ] = setup_project_ht_WUSTL;

% 1. Retrieval
ht_retrieval;

% 2. Geometric verification
ht_top100_sparsePE_localization;

% 3. Pose verification
ImgList_densePE = ImgList_sparsePE; % Force dense PV to use sparse PE results.
ht_top10_densePV_localization;
