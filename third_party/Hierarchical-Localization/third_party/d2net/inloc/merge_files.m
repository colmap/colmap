function ImgList = merge_files(file1, file2)
    f1 = load(file1);
    ImgList_file1 = f1.ImgList;
    f2 = load(file2);
    ImgList_file2 = f2.ImgList;

    PV_topN = 10;

    n1 = 0;
    n2 = 0;
    ImgList = struct('queryname', {}, 'topNname', {}, 'topNscore', {}, 'P', {});
    for ii = 1:1:length(ImgList_file1)
        ImgList(ii).queryname = ImgList_file1(ii).queryname;
        
        sum_scores = containers.Map('KeyType', 'char', 'ValueType', 'double');
        for jj = 1 : PV_topN
            name = char(ImgList_file1(ii).topNname(jj));
            if isKey(sum_scores, name)
                sum_scores(name) = sum_scores(name) + ImgList_file1(ii).topNscore(jj);
            else
                sum_scores(name) = ImgList_file1(ii).topNscore(jj);
            end
            name = char(ImgList_file2(ii).topNname(jj));
            if isKey(sum_scores, name)
                sum_scores(name) = sum_scores(name) + ImgList_file2(ii).topNscore(jj);
            else
                sum_scores(name) = ImgList_file2(ii).topNscore(jj);
            end
        end
        
        max_score = 0;
        img_name = 0;
        for key = keys(sum_scores)
            if sum_scores(char(key)) > max_score
                max_score = sum_scores(char(key));
                img_name = key;
            end
        end
        
        id_dense = 0;
        id_sparse = 0;
        for jj = 1 : PV_topN
            if strcmp(char(ImgList_file1(ii).topNname(jj)), img_name)
                id_dense = jj;
            end
            if strcmp(char(ImgList_file2(ii).topNname(jj)), img_name)
                id_sparse = jj;
            end
        end
        
        if id_sparse == 0
            n1 = n1 + 1;
            ImgList(ii).topNscore = [ImgList_file1(ii).topNscore(id_dense)];
            ImgList(ii).topNname = [ImgList_file1(ii).topNname(id_dense)];
            ImgList(ii).P = [ImgList_file1(ii).P(id_dense)];
            continue
        end
        
        if id_dense == 0
            n2 = n2 + 1;
            ImgList(ii).topNscore = [ImgList_file2(ii).topNscore(id_sparse)];
            ImgList(ii).topNname = [ImgList_file2(ii).topNname(id_sparse)];
            ImgList(ii).P = [ImgList_file2(ii).P(id_sparse)];
            continue
        end
        
        max_score = 0;
        if ImgList_file1(ii).topNscore(id_dense) > ImgList_file2(ii).topNscore(id_sparse)
            n1 = n1 + 1;
            ImgList(ii).topNscore = [ImgList_file1(ii).topNscore(id_dense)];
            ImgList(ii).topNname = [ImgList_file1(ii).topNname(id_dense)];
            ImgList(ii).P = [ImgList_file1(ii).P(id_dense)];
        else
            n2 = n2 + 1;
            ImgList(ii).topNscore = [ImgList_file2(ii).topNscore(id_sparse)];
            ImgList(ii).topNname = [ImgList_file2(ii).topNname(id_sparse)];
            ImgList(ii).P = [ImgList_file2(ii).P(id_sparse)];
        end
    end

    fprintf(1, "%d file 1 poses & %d file 2 poses selected\n", n1, n2);
end