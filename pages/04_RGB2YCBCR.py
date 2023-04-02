# Author: - Achmadya Ridwan Ilyawan
#         - Fahmi Ahmad Fadilah
#         - Hilman Permana
# Date: 2023-04-02
# Description: Konversi warna citra dari RGB ke YCbCr

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# konversi warna RGB ke YCbCr
def RGB2YCBCR(I):
        Ir = I[:,:,0]
        Ig = I[:,:,1]
        Ib = I[:,:,2]
        [m,n] = Ir.shape
        
        # konversi RGB ke YCbCr
        for i in range(m):
            for j in range(n):
                Y = (0.299*Ir[i,j]) + (0.587*Ig[i,j]) + (0.114*Ib[i,j])
                Cb = 128 - (0.168736*Ir[i,j]) - (0.331264*Ig[i,j]) + (0.5*Ib[i,j])
                Cr = 128 + (0.5*Ir[i,j]) - (0.418688*Ig[i,j]) - (0.081312*Ib[i,j])
                
                Ir[i,j] = Y
                Ig[i,j] = Cb
                Ib[i,j] = Cr
        
        I = cv2.merge((Ir,Ig,Ib))
        return I


# judul halaman web
st.title('Konversi Warna Citra dari RGB ke YCbCr')

# upload gambar
uploaded_file = st.file_uploader("Upload Files", type=["jpg", "png", "jpeg"])

# menampilkan gambar
if uploaded_file is not None:

    image = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # menampilkan gambar
    colimg1, colimg2 = st.columns(2)

    with colimg1:
        st.image(image, caption='Gambar Asli', use_column_width=True)

    # konversi warna
    with colimg2:
        image = RGB2YCBCR(image)
        st.image(image, caption='Gambar Hasil', use_column_width=True)



# menampilkan kode sumber
tab1, tab2 = st.tabs(['MATLAB', 'Python'])

with tab1:
    st.markdown('''
    ```matlab
    function f = RGB2YCBCR(I)
        Ir = I(:,:,1);
        Ig = I(:,:,2);
        Ib = I(:,:,3);
        [m,n] = size(Ir);
        
        k = [0;128;128];
        l = [0.299 0.587 0.114;
            -0.169 -0.331 0.500;
            0.500 -0.419 -0.081;];
        
        for i = 1 : m
        for j = 1 : n
            rgb = [Ir(i,j); Ig(i,j); Ib(i,j)];
            ycbcr = k+l*double(rgb);
            Iy(i,j)  = double(ycbcr(1,:));
            Icb(i,j) = double(ycbcr(2,:));
            Icr(i,j) = double(ycbcr(3,:));
        end
        end
        Iycbcr(:,:,1) = Iy;
        Iycbcr(:,:,2) = Icb;
        Iycbcr(:,:,3) = Icr;
        
        % figure(1), imshow(Iycbcr);
        f = Iycbcr;
    end
    ```
    ''', unsafe_allow_html=True)


with tab2:
    st.markdown('''
    ```python
    def RGB2YCBCR(I):
        Ir = I[:,:,0]
        Ig = I[:,:,1]
        Ib = I[:,:,2]
        [m,n] = Ir.shape
        
        # konversi RGB ke YCbCr
        for i in range(m):
            for j in range(n):
                Y = (0.299*Ir[i,j]) + (0.587*Ig[i,j]) + (0.114*Ib[i,j])
                Cb = 128 - (0.168736*Ir[i,j]) - (0.331264*Ig[i,j]) + (0.5*Ib[i,j])
                Cr = 128 + (0.5*Ir[i,j]) - (0.418688*Ig[i,j]) - (0.081312*Ib[i,j])
                
                Ir[i,j] = Y
                Ig[i,j] = Cb
                Ib[i,j] = Cr
        
        I = cv2.merge((Ir,Ig,Ib))
        return I
    ```
    ''', unsafe_allow_html=True)