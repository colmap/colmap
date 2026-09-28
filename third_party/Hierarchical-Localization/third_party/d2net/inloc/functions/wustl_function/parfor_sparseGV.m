function parfor_sparseGV( qname, dbname, params )


[~, dbbasename, ~] = fileparts(dbname);
this_sparsegv_matname = fullfile(params.output.gv_sparse.dir, qname, [dbbasename, params.output.gv_sparse.matformat]);

if exist(this_sparsegv_matname, 'file') ~= 2
    %load features
    qfmatname = fullfile(params.input.feature.dir, params.data.q.dir, [qname, params.input.feature.q_sps_matformat]);
    if exist(qfmatname, 'file') ~= 2
        Iqname = fullfile(params.data.dir, params.data.q.dir, qname);
        [f, d] = features_WUSTL(Iqname);
        [qfdir, ~, ~] = fileparts(qfmatname);
        if exist(qfdir, 'dir') ~= 7
            mkdir(qfdir);
        end
        save('-v6', qfmatname, 'f', 'd');
    end
    features_q = load(qfmatname);
    
    dbfmatname = fullfile(params.input.feature.dir, params.data.db.cutout.dir, [dbname, params.input.feature.db_sps_matformat]);
    if exist(dbfmatname, 'file') ~= 2
        Idbname = fullfile(params.data.dir, params.data.db.cutout.dir, dbname);
        [f, d] = features_WUSTL(Idbname);
        [dbfdir, ~, ~] = fileparts(dbfmatname);
        if exist(dbfdir, 'dir') ~= 7
            mkdir(dbfdir);
        end
        save('-v6', dbfmatname, 'f', 'd');
    end
    features_db = load(dbfmatname);
    
    %geometric verification
    if size(features_db.d, 2) < 6
        H = nan(3, 3);
        inls_qidx = [];
        inls_dbidx = [];
        inliernum = 0;
        matches = [];
        inliers = [];
    else
        
        %geometric verification (homography lo-ransac)
        [matches, inliers, H, ~] = at_sparseransac(features_q.f,features_q.d,features_db.f,features_db.d,3,10);
        inliernum = length(inliers);
        inls_qidx = inliers(1, :); inls_dbidx = inliers(2, :);
    end
    
    %save
    if exist(fullfile(params.output.gv_sparse.dir, qname), 'dir') ~= 7
        mkdir(fullfile(params.output.gv_sparse.dir, qname));
    end
    save('-v6', this_sparsegv_matname, 'H', 'inliernum', 'inls_qidx', 'inls_dbidx', 'matches', 'inliers');
    
%     %debug
%     Iq = imread(fullfile(params.data.dir, params.data.q.dir, qname));
%     Idb = imread(fullfile(params.data.dir, params.data.db.cutout.dir, dbname));
%     figure();
%     ultimateSubplot ( 2, 1, 1, 1, 0.01, 0.05 );
%     imshow(rgb2gray(Iq));hold on;
%     plot(features_q.f(1, inls_qidx), features_q.f(2, inls_qidx),'g.');
%     ultimateSubplot ( 2, 1, 2, 1, 0.01, 0.05 );
%     imshow(rgb2gray(Idb));hold on;
%     plot(features_db.f(1, inls_dbidx), features_db.f(2, inls_dbidx),'g.');
% 
%     keyboard;
    
end



end

