im = imread('images/Transparent_Objects/wine/DSC_0480.JPG');
gray = imresize(rgb2gray(im), 0.25);
% gray = gray(200:800,600:1000);
bw = edge(gray, 'Prewitt', 0.05, 'nothinning');
% bw = imfill(bw,'holes');
figure;
gray = 0.5*gray+uint8(255*bw);
imshow(gray)

% figure;
% g_high = impulsegaussian2d(100, 40);
% filtered_im2 = conv2(gray, g_high, 'same');
% filtered_im2 = imbinarize(filtered_im2*0.1 + bw);
% cc = bwconncomp(filtered_im2);
% numPixels = cellfun(@numel,cc.PixelIdxList);
% [biggest,idx] = max(numPixels);
% filtered_im2 = zeros(size(filtered_im2));
% filtered_im2(cc.PixelIdxList{idx}) = 0;
% filtered_im2 = imfill(filtered_im2,'holes');

% imshow(filtered_im2)
bw = edge(gray, 'prewitt', 'nothinning');
bw = imfill(bw,'holes');
figure;
imshow(bw);

% imagesc(log(abs(fftshift(fft2(filtered_im2))))), axis image, colormap gray
% imshow(y);

function f = gaussian2d(N,sigma)
    % N is grid size, sigma is std dev
    [x,y] = meshgrid(round(-N/2):round(N/2), round(-N/2):round(N/2));
    f = exp(-x.^2/(2*sigma^2)-y.^2/(2*sigma^2));
    f = f./sum(f(:));
end

function f = impulsegaussian2d(N,sigma)
    % N is grid size, sigma is std dev
    [x,y] = meshgrid(round(-N/2):round(N/2), round(-N/2):round(N/2));
    f = exp(-x.^2/(2*sigma^2)-y.^2/(2*sigma^2));
    f = -f./sum(f(:));
    f(round(N/2),round(N/2)) = f(round(N/2),round(N/2)) + 1;
end