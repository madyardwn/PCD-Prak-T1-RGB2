
% Nama File: RGB2HSI.m
% Deskripsi: Melakukan konversi model warna citra dari RGB ke HSI
 
% Inisialsisasi
function f = RGB2HSI(I)
	Ir = double(I(:,:,1));
	Ig = double(I(:,:,2));
	Ib = double(I(:,:,3));
	[m,n] = size(Ir);

	for i = 1 : m
	   for j = 1 : n
		   r = Ir(i,j)/255;
		   g = Ig(i,j)/255;
		   b = Ib(i,j)/255;
		   
		   minrgb = min(r,min(g,b));
		   i2 = (r+g+b)/3;
		   if r==g && g==b 
				h = 0;
				s = 0;
		   else
		   
			   a = (r-g)+(r-b);
			   rg2 = (r-g)*(r-g);
			   rb2 = (r-b)*(g-b);
			   b2 = sqrt(double(rg2+rb2));
			   alpha = acos(double(0.5*(a/b2))) * 180/pi;
			   
			   if g >= b
				   h = alpha;
			   else
				   h = 360 - alpha;   
			   end
			   
			   s = 1 - (3*(minrgb/(r+g+b)));
		   end
		   Ih(i,j) = double(h/360);
		   Is(i,j) = double(s);
		   Ii(i,j) = double(i2);
	   end
	end
	Ihsi(:,:,1) = Ih;
	Ihsi(:,:,2) = Is;
	Ihsi(:,:,3) = Ii;
	 
	% figure(1), imshow(Ihsi);
    f = Ihsi;
end

