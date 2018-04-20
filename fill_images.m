for i=1:50
   name = ['Dataset/datam_', num2str(i), '.jpg'];
   im = imread(name);
   im = imbinarize(im);
   im = imfill(im,'holes');
   imwrite(im, ['Dataset/Dataf_', num2str(i), '.jpg']);
end