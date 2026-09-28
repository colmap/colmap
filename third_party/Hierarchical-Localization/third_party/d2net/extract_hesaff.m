fid = fopen('image_list_hpatches_sequences.txt');

tline = fgetl(fid);
while ischar(tline)
    disp(tline);
    I = im2single(imread(tline));
    if size(I, 3) > 1
        I = rgb2gray(I);
    end
    
    [F, D, info] = vl_covdet(I, 'Method', 'Hessian', ...
                                'EstimateAffineShape', true, ...
                                'EstimateOrientation', true, ...
                                'DoubleImage', false, ...
                                'peakThreshold', 14 / 256^2);
    keypoints = F';
    scores = info.peakScores;
    descriptors = D';
    
    save([tline '.hesaff'], 'keypoints', 'scores', 'descriptors');
    
    tline = fgetl(fid);
end

fclose(fid);
