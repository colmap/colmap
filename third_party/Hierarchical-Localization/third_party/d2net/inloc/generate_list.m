startup;
params = setup_project;

ht_retrieval;

shortlist_topN = 100;

query_dir = fullfile(params.data.dir, params.data.q.dir);
db_dir = fullfile(params.data.dir, params.data.db.cutout.dir);

image_list_file = fopen('image_list.txt', 'w');

for ii = 1:1:length(ImgList_original)
    query_image_path = [query_dir '/' ImgList_original(ii).queryname];

    fprintf(image_list_file, '%s\n', query_image_path);
    
    for jj = 1:1:shortlist_topN
        db_image_path = [db_dir '/' ImgList_original(ii).topNname{jj}];

        fprintf(image_list_file, '%s\n', db_image_path);
    end
end

fclose(image_list_file);
