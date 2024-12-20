% READING INPUT IMAGE 
img = imread('EYE.jpg'); % Read input image

% EXTRACTING GREEN CHANNEL OF IMAGE
a = zeros(size(img, 1), size(img, 2));
green = img(:,:,2); % Green channel
test_image = cat(3, a, green, a);
figure, imshow(test_image), title('Green channel') 
 
% RESIZING IMAGE: we resized the image to 584 X 565 pixels to make commputation faster as orignal was very big
resized_image = imresize(test_image, [584 565]); 
figure, imshow(resized_image), title('Resized green channel image')
 
%Pixel values are in integer they cann't accept decimal points but as we
%are going to segment blood vssels we need precesion till decimal points so we convert image into double data type 
converted_image = im2double(resized_image);
%Converting into CIE lab colour space it was defined by International
%commission of illumination in 1976, 
lab_image = rgb2lab(converted_image);  %converts sRGB values to CIE 1976 L*a*b* values.  
figure, imshow(lab_image)

% 3 dimension of contatinated array image is in CIE format
fill = cat(3, 1,0,0); %red channel remain as prominent
filled_image = bsxfun(@times, fill, lab_image); %element wise binary operation 
figure, imshow(filled_image)
title('filled image')

%RESHAPING FILLED IMAGE
reshaped_lab_image = reshape(filled_image, [], 3); %no particular dimension neded so bllank [] 3 is existing diension of filled image
[C, S] = pca(reshaped_lab_image); %applying principal component analysis  it returns cofeccient and score of principle compoonent
S = reshape(S, size(lab_image)); %reshaping score based on size of lab image
S = S(:, :, 1); %we need all rows and colum of first channel 
gray_image = (S-min(S(:)))./(max(S(:))-min(S(:)));
figure, imshow(gray_image)
title('gray image')

enhanced_image = adapthisteq(gray_image, 'numtiles', [8 8], 'nBins', 128);
figure, imshow(enhanced_image)
title('Enhanced image')

% creating eye mask
eyemask = imbinarize(enhanced_image, graythresh(enhanced_image));
eyemask = imfill(eyemask, 'holes');
eyemask = bwareaopen(eyemask,100);
imshow(eyemask);
title('eye mask')

%applying mask to eluminate background noise
enhanced_image(~eyemask)= 0;
figure, imshow(enhanced_image)
title('new enhanced image')

avg_filter = fspecial('average', [10 10]);
filtered_image = imfilter(enhanced_image, avg_filter);
figure, imshow(filtered_image)
title('filtered image')

subtrated_image = imsubtract(filtered_image, enhanced_image);
level = threshold_level(subtrated_image);
figure, imshow(filtered_image)
title('level image')

binary_image = imbinarize(subtrated_image, level-0.008);
figure, imshow(binary_image)
title('Binary image')

se = strel('disk',1);
afterOpening = imopen(binary_image,se);
figure
imshow(afterOpening,[]);
title('opening')

clean_image = bwareaopen(afterOpening, 90);
figure, imshow(clean_image)
title('Clean Image')

%remove shorter comings from vessels
clean_vessel = bwmorph(clean_image,'spur',5);
figure
imshow(clean_vessel)
title('Shot Vessels Removed')

Complemented_image = imcomplement(clean_vessel);
figure, imshow(Complemented_image)
title('Complemented Image')

Final_result = colorize_image(resized_image, Complemented_image, [0 0 0]);
figure, imshow(Final_result)
title('Coloured Image')


%DETECTING EDGES 

%using canny edge detection
BW2 = edge(clean_vessel,'Canny');
figure
imshow(BW2)
title('Canny Edge Detection')

% Prewitt = edge(clean_vessel,'Prewitt');
% figure
% imshow(Prewitt)
% title('Prewitt Edge Detection')
% 
% Roberts = edge(clean_vessel,'Roberts');
% figure
% imshow(Roberts)
% title('Roberts Edge Detection')
% 
% log = edge(clean_vessel,'log');
% figure
% imshow(log)
% title('log Edge Detection')
% 
% zerocross = edge(clean_vessel,'zerocross');
% figure
% imshow(zerocross)
% title('zerocross Edge Detection')
% 
% approxcanny = edge(clean_vessel,'approxcanny');
% figure
% imshow(approxcanny)
% title('approxcanny Edge Detection')
% 
% Sobel = edge(clean_vessel,'Sobel');
% figure
% imshow(Sobel)
% title('approxcanny Edge Detection')
% 
% figure 
% montage({BW2,Prewitt,Roberts,log,zerocross,approxcanny,Sobel})


%removing outer ring
clean_vessel(~eyemask)= 0;
%figure
%imshow(clean_vessel);
%title('Without Outer Ring')

%centerline of vessels (Skeletonization)
centerline_skleton = bwmorph(clean_vessel,'skel',Inf);
% figure
% imshow(centerline_skleton)
% title('Centerline of Vessels')

%overlaying centerline over segmented vessels
overlayed = imoverlay(clean_vessel, centerline_skleton, 'r');
figure
imshow(overlayed)
title('Overlayed Image with Centerline')

%finding branchpoints and dialting them
Branchpoint = bwmorph(centerline_skleton,'branchpoints');
se = strel('disk',4,4);
dialted_branchpoint = imdilate(Branchpoint,se);
figure
imshow(dialted_branchpoint)
title('Dialted Branchpoint')

overlayed = imoverlay(clean_vessel, dialted_branchpoint, 'r');
figure
imshow(overlayed)
title('Branchpoint on Vessels')

%Image Without Branchpoints
%skleton image without branchpoint
centerline_skleton = centerline_skleton & ~dialted_branchpoint;
figure
imshow(centerline_skleton);
title('Skleton Image Without Branchpoint')

%discarding shorter segments(manual value selection)
minlength = 15;
vessel_segments = bwareaopen(centerline_skleton, minlength, 8);
figure
imshow(vessel_segments);
title('Discarding Shorter Segments')

%for understanding segments of which vessel is there
overlayinggray = imoverlay(enhanced_image, centerline_skleton, 'r');
figure
imshow(overlayinggray)
title('Overlayed Image with Centerline on Enhancd Vessel image')

%labelling and dilating L = labelled 
%set(gca, 'xlimmode','auto','ylimmode','auto')
L = bwlabel(vessel_segments,8);
%imshow(label2rgb(imdilate(L,strel('disk',4))jet(max(L(:))),'k','shuffle'));
RGB = label2rgb(L,jet(max(L(:))),'k','shuffle'); 
se2 = strel('disk',1);
dialted_labelled_branch = imdilate(RGB,se2);
figure
imshow(dialted_labelled_branch)
title('Dialted Labelled Branch')

% [r, c] = find(L==2);
% rc = [r c]

imtool(dialted_labelled_branch)

%  figure
%  imshow(dialted_labelled_branch == 3);
%  title('Dialted Labelled Branch')

%    figure
%    h = imshow(L == 1);
%    title('play')
%    for ii = 1:max(L(:))
%        %dialted_labelled_branch = ii;
%        set(h,'cdata',L ==ii)
%        pause
%    end

figure 
imshow(L == 25);
set(gca,'xlim',[320,380],'ylim',[412,465]);
title('labelled vessel no popos up')

%endponts way1
[row, col] = find(bwmorph(L == 25, 'endpoints'));
hold on 
plot(col, row,'r.','MarkerSize',36)

% plot(col, row,'-o','MarkerSize',36)
% plot(col,row,'--gs',...
%     'LineWidth',2,...
%     'MarkerSize',10,...
%     'MarkerEdgeColor','b',...
%     'MarkerFaceColor',[0.5,0.5,0.5])

%endpoints way2
endpoint_fcn = @(nhood) (nhood(5)==1)&&(nnz(nhood)==2);

figure 
imshow(L == 25);
set(gca,'xlim',[320,380],'ylim',[412,465]);
%L=25 X =  338 427 Y = 360 451 rect1 = 336.5 425.5 3 3 rect2 = 358.5 449.5 3 3
%rect(1) = rectangle('position',[336.5 425.5 3 3],'edgecolor', 'red');
%rect(1) = rectangle('position',[358.5 449.5 3 3],'edgecolor', 'red');

%cross check
endpoint_fcn([0 0 0;0 1 0; 0 1 1])

endpoint_fcn([0 0 0;0 1 0; 0 0 1])

endpoint_lut = makelut(endpoint_fcn, 3);

%set(gca,'xlim',[320,380],'ylim',[412,465])

%plot(col, row,'r.','MarkerSize',36)

[row, col] = find(bwlookup(L == 25, endpoint_lut));
hold on
plot(col, row,'yo','MarkerSize',36)
plot(col, row,'r.','MarkerSize',36)

%MEASURING CURVE LENGTH WAY1 : perimeter
set(gca, 'xlimmode','auto','ylimmode','auto')
L = bwlabel(vessel_segments,8);
%imshow(label2rgb(imdilate(L,strel('disk',4))jet(max(L(:))),'k','shuffle'));
RGB = label2rgb(L,jet(max(L(:))),'k','shuffle'); 
se2 = strel('disk',1);
dialted_labelled_branch = imdilate(RGB,se2);
figure
imshow(dialted_labelled_branch)
title('Arclength via perimeter')
hold on
stats = regionprops(L,'Centroid','Perimeter');
delete(findobj(gcf,'tag','mybalel'));

pathlengths = [stats.Perimeter]/2;
for ii = 1:numel(stats)
    text(stats(ii).Centroid(1),stats(ii).Centroid(2),sprintf('*%0.1f pixels',pathlengths(ii)),'fontunits','normal','color',[0.9 0.7 0],'tag','mylabel')
end
title('segments length')

%WAY 2:  using geodisc dist transform
figure
im = L == 25;
imshow(im);
set(gca,'xlim',[320,380],'ylim',[412,465]);
%im = imcrop(L,[320 412 380 465]);%330 420 34 35
%imshow(im);
[row, col] = find(bwlookup(im, endpoint_lut));
hold on
plot(col, row,'m.','MarkerSize',36)
title('single vessel')

gdist = bwdistgeodesic(im, col(1), row(1), 'quasi-euclidean');
imshow(gdist,[])
title('geodisc distance transform')
hold on
text(col(1)+2, row(1)-1, 'starting point','color','r')
plot(col(1), row(1), 'm.', 'markersize', 36)
colormap([0 0 0;jet])
colorbar('horizontal')
 
%using geodesic and using lookup table
figure
imshow(L);
delete(findobj(gcf,'tag','myLabel'));
hold on 

mydist = @(p1,p2) sqrt((p1(1)-p1(2))^2 + (p2(1)-p2(2))^2);
stats = regionprops(L,'Centroid');

%preallocate
dists = zeros(numel(stats,1));
pathlengths = dists;

for ii = 1:numel(stats)
    tmpIm = L == ii;
    %[r,c] = find(bwmorph(tmpIm,'endpoints'));
    [r,c] = find(bwlookup(tmpIm,endpoint_lut));
    hold on
    plot(c,r,'r.','markersize',12,'tag','mylabel')
    plot(c(1),r(1),'go','markersize',10,'tag','mylabel')
    try
        %calc endpoint dist
        dists(ii) = mydist(r,c);
        %calc pathlengths
        pathlengths(ii)= max(max(bwdistgeodesic(tmpIm,col(1), row(1),'quasi-euclidean')));
        text(stats(ii).Centroid(1),stats(ii).Centroid(2),sprintf('*%0.1f pixels',dists(ii)),'fontsize',9,'fontweight','normal','color',[0.7 0.7 0],'tag','mylabel');
    catch
        %fprintf('didnt work')
        dists(ii) = mydist(r,c);
        pathlengths(ii)= max(max(bwdistgeodesic(tmpIm,col(1), row(1),'quasi-euclidean')));
        text(stats(ii).Centroid(1),stats(ii).Centroid(2),sprintf('*%0.1f pixels',dists(ii)),'fontsize',9,'fontweight','normal','color',[0.7 0.7 0],'tag','mylabel');
    end
end        
      
% TURTUOSITIES
imshow(L)
delete(findobj(gcf,'tag','mylabel'));
hold on
turtuosity = pathlengths./dists; %pathlengths./dists
colors = [0.9 0.7 0;1 0 0];
criticalvalue = 1.2;
for ii = 1:numel(dists)
    %fprintf('didnt work')1+(turtuosity(ii)>criticalvalue)*2,:
    text(stats(ii).Centroid(1),stats(ii).Centroid(2),sprintf('*V%d: %0.2f',ii,turtuosity(ii)),'fontsize',9+(turtuosity(ii)>criticalvalue)*2,'fontweight','b','color',colors(1+(turtuosity(ii)>criticalvalue),:),'tag','myLabel')
end    
title('Turtuosity','color',[0 0.7 0],'fontsize',10)


%Complemented_image = imcomplement(clean_vessel);
%figure, imshow(Complemented_image)
%title('Complemented Image')

%figure 
%montage({test_image,resized_image,enhanced_image,filtered_image,binary_image,afterOpening, clean_image, BW2, BW3,Complemented_image })

%Final_result = colorize_image(resized_image, Complemented_image, [0 0 0]);
%figure, imshow(Final_result)
%title('Coloured Image')

