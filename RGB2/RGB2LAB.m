% Nama File: RGB2LAB.m
% Deskripsi: Melakukan konversi model warna citra dari RGB ke CIELab

% Inisialisasi
function f = RGB2LAB(I)
	Ixyz = rgb2xyz(I);
	Ix = Ixyz(:,:,1);
	Iy = Ixyz(:,:,2);
	Iz = Ixyz(:,:,3);
	[m,n] = size(Ix);
	 
	% white point d65
	xn = 0.95047;
	yn = 1;
	zn = 1.08883;

	for i = 1 : m
	   for j = 1 : n
		   IL(i,j) = 116*fLab(Iy(i,j)/yn)-16;
		   Ia(i,j) = 500*(fLab(Ix(i,j)/xn)-fLab(Iy(i,j)/yn));
		   Ib(i,j) = 200*(fLab(Iy(i,j)/yn)-fLab(Iz(i,j)/zn));
	   end
	end
	ILab(:,:,1) = IL;
	ILab(:,:,2) = Ia;
	ILab(:,:,3) = Ib;
	 
	% figure(1), imshow(ILab);
    f = ILab;
end