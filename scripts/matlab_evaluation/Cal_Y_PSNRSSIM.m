function [psnr_cur, ssim_cur] = Cal_Y_PSNRSSIM(A,B,row,col)
    % shave border if needed
    if nargin > 2
        [n,m,~]=size(A);
        A = A(row+1:n-row,col+1:m-col,:);
        B = B(row+1:n-row,col+1:m-col,:);
    end
    % RGB --> YCbCr
    if 3 == size(A, 3)
        A = rgb2ycbcr(A);
        A = A(:,:,1);
    end
    if 3 == size(B, 3)
        B = rgb2ycbcr(B);
        B = B(:,:,1);
    end
    % calculate PSNR
    A=double(A); % Ground-truth
    B=double(B); %
    
    e=A(:)-B(:);
    mse=mean(e.^2);
    psnr_cur=10*log10(255^2/mse);
    
    % calculate SSIM
    [ssim_cur, ~] = ssim_index(A, B);
end
