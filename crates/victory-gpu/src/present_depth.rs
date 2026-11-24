//! Uploads a live diagnostics scene preview to the window's back buffer.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
};

use crate::capture::ScenePreview;
use crate::host::Host;

pub(crate) struct PreviewUpload {
    pub(crate) frame: u64,
    pub(crate) color: bool,
    buffer: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

impl PreviewUpload {
    pub(crate) fn new(
        host: &Host,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
        image: &ScenePreview,
    ) -> Option<Self> {
        if !matches!(
            format,
            DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM
        ) || width == 0
            || height == 0
            || image.width == 0
            || image.height == 0
        {
            return None;
        }
        let src_row = (image.width as usize).checked_mul(4)?;
        if src_row.checked_mul(image.height as usize)? != image.rgba.len() {
            return None;
        }
        let row = width.checked_mul(4)?;
        let pitch = row
            .div_ceil(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
            .checked_mul(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)?;
        let len = (pitch as usize).checked_mul(height as usize)?;
        let (buffer, data) = host.upload_buffer(len as u64)?;
        // SAFETY: the new upload buffer is mapped for writes, has `len`
        // bytes, and no GPU command references it yet.
        let destination = unsafe { std::slice::from_raw_parts_mut(data, len) };
        scale_rgba(
            image,
            destination,
            width as usize,
            height as usize,
            pitch as usize,
            format == DXGI_FORMAT_B8G8R8A8_UNORM,
        );
        Some(Self {
            frame: image.frame,
            color: image.color,
            buffer,
            footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: format,
                    Width: width,
                    Height: height,
                    Depth: 1,
                    RowPitch: pitch,
                },
            },
        })
    }

    /// Copy the uploaded RGBA8 or BGRA8 pixels into a copy-dest back buffer.
    pub(crate) fn copy_to(&self, list: &ID3D12GraphicsCommandList, back: &ID3D12Resource) {
        // SAFETY: the locations borrow the live resources. D3D12's copy
        // location suppresses dropping its embedded resource pointer.
        unsafe {
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: self.footprint,
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
    }
}

fn scale_rgba(
    image: &ScenePreview,
    out: &mut [u8],
    width: usize,
    height: usize,
    pitch: usize,
    bgra: bool,
) {
    let (source_width, source_height) = (image.width as usize, image.height as usize);
    let columns: Vec<_> = (0..width).map(|x| 4 * (x * source_width / width)).collect();
    for (y, row) in out.chunks_exact_mut(pitch).enumerate() {
        let from = (y * source_height / height) * source_width * 4;
        let source = &image.rgba[from..from + source_width * 4];
        for (pixel, &column) in row[..width * 4].chunks_exact_mut(4).zip(&columns) {
            let (r, g, b) = (source[column], source[column + 1], source[column + 2]);
            let rgb = if bgra { [b, g, r] } else { [r, g, b] };
            pixel[..3].copy_from_slice(&rgb);
            pixel[3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preview_scales_into_padded_rows() {
        let image = ScenePreview {
            frame: 1,
            width: 2,
            height: 1,
            rgba: vec![30, 40, 50, 255, 170, 160, 150, 255],
            color: true,
        };
        let mut bytes = [0xff; 48];
        scale_rgba(&image, &mut bytes, 4, 2, 24, false);
        for row in bytes.chunks_exact(24) {
            assert_eq!(&row[..4], &[30, 40, 50, 255]);
            assert_eq!(&row[4..8], &row[..4]);
            assert_eq!(&row[8..12], &[170, 160, 150, 255]);
            assert_eq!(&row[12..16], &row[8..12]);
            assert_eq!(&row[16..], &[0xff; 8]);
        }
        scale_rgba(&image, &mut bytes, 4, 2, 24, true);
        assert_eq!(&bytes[..4], &[50, 40, 30, 255]);
        assert_eq!(&bytes[8..12], &[150, 160, 170, 255]);
    }
}
