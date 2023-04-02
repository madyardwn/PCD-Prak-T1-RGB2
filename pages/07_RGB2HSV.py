# Author: - Achmadya Ridwan Ilyawan
#         - Fahmi Ahmad Fadilah
#         - Hilman Permana
# Date: 2023-04-02
# Description: Konversi warna citra dari RGB ke HSV

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# konversi warna RGB ke HSV
def RGB2HSV(I):
    Ir = I[:, :, 0].astype(float) / 255
    Ig = I[:, :, 1].astype(float) / 255
    Ib = I[:, :, 2].astype(float) / 255

    m, n = Ir.shape
    Iv = np.zeros((m, n))
    Is = np.zeros((m, n))
    Ih = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            r = Ir[i, j]
            g = Ig[i, j]
            b = Ib[i, j]

            v = max(max(r, g), b)
            vm = v - min(min(r, g), b)
            if v == 0:
                s = 0
            else:
                s = vm / v
            if s == 0:
                h = 0
            elif v == r:
                h = 60 * (g - b) / vm
            elif v == g:
                h = 120 + 60 * (b - r) / vm
            elif v == b:
                h = 240 + 60 * (r - g) / vm
            if h < 0:
                h += 360

            Ih[i, j] = h / 360
            Is[i, j] = s
            Iv[i, j] = v

    Ihsv = np.zeros((m, n, 3))
    Ihsv[:, :, 0] = Ih
    Ihsv[:, :, 1] = Is
    Ihsv[:, :, 2] = Iv

    return Ihsv


# judul halaman web
st.title('Konversi Warna Citra dari RGB ke HSV')

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
        image = RGB2HSV(image)
        st.image(image, caption='Gambar Hasil', use_column_width=True)



# menampilkan kode sumber
tab1, tab2 = st.tabs(['MATLAB', 'Python'])

with tab1:
    st.markdown('''
    ```matlab
    function f = RGB2HSV(I)
        Ir = double(I(:,:,1));
        Ig = double(I(:,:,2));
        Ib = double(I(:,:,3));
        [m,n] = size(Ir);
        
        for i = 1 : m
        for j = 1 : n
            r = Ir(i,j)/255;
            g = Ig(i,j)/255;
            b = Ib(i,j)/255;
            
            v = max(max(r,g),b);
            vm = v-min(r,min(g,b));
            if v==0
                s = 0; 
            elseif v>0
                s = vm/v;
            end
            if s==0
                h=0;
            elseif v==r
                h=60/360*(mod((g-b)/vm,6));
            elseif v==g
                h=60/360*(2+((b-r)/vm));
            elseif v==b
                h=60/360*(4+((r-g)/vm));
            end   
            Iv(i,j) = v;
            Is(i,j) = s;
            Ih(i,j) = h;
        end
        end
        Ihsv(:,:,1) = Ih;
        Ihsv(:,:,2) = Is;
        Ihsv(:,:,3) = Iv;
        
        % figure(1), imshow(Ihsv);
        f = Ihsv;
    end

    ```
    ''', unsafe_allow_html=True)


with tab2:
    st.markdown('''
    ```python
    def RGB2HSV(I):
        Ir = I[:, :, 0].astype(float) / 255
        Ig = I[:, :, 1].astype(float) / 255
        Ib = I[:, :, 2].astype(float) / 255

        m, n = Ir.shape
        Iv = np.zeros((m, n))
        Is = np.zeros((m, n))
        Ih = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                r = Ir[i, j]
                g = Ig[i, j]
                b = Ib[i, j]

                v = max(max(r, g), b)
                vm = v - min(min(r, g), b)
                if v == 0:
                    s = 0
                else:
                    s = vm / v
                if s == 0:
                    h = 0
                elif v == r:
                    h = 60 * (g - b) / vm
                elif v == g:
                    h = 120 + 60 * (b - r) / vm
                elif v == b:
                    h = 240 + 60 * (r - g) / vm
                if h < 0:
                    h += 360

                Ih[i, j] = h / 360
                Is[i, j] = s
                Iv[i, j] = v

        Ihsv = np.zeros((m, n, 3))
        Ihsv[:, :, 0] = Ih
        Ihsv[:, :, 1] = Is
        Ihsv[:, :, 2] = Iv

        return Ihsv
    ```
    ''', unsafe_allow_html=True)