//! 索引序列化格式（含版本管理）
//!
//! 设计文档第二层 F.1：
//! 文件头布局（固定 16 字节，little-endian）：
//!   [0..4]   magic:   0x414E4E58 ("ANNX")
//!   [4..8]   version: u32，格式变更时递增，当前为 1
//!   [8..12]  flags:   u32，预留扩展位
//!   [12..16] crc32:   u32，文件体校验和
//!
//! 加载时：
//!   版本号不匹配 → 报错并提示重建索引，不静默加载
//!   校验和不一致 → 报错，拒绝加载
//!   跨架构加载须做字节序转换

use thiserror::Error;

/// 文件头 magic 值："ANNX" 的 little-endian 表示
/// 设计文档：0x414E4E58
pub const INDEX_MAGIC: u32 = 0x414E4E58;

/// 当前序列化格式版本号
/// 设计文档：格式变更时递增，当前为 1
pub const INDEX_VERSION: u32 = 1;

/// flags bit 0：文件体包含 BuildMetadata trailer（设计文档 F.7）
pub const FLAG_HAS_METADATA: u32 = 0x1;

/// flags bit 1：文件体包含 LayeredNavigation trailer（设计文档：随机层级导航）
pub const FLAG_HAS_LAYERED_NAV: u32 = 0x2;

/// 文件头大小（字节）
pub const HEADER_SIZE: usize = 16;

/// 序列化错误类型
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum SerializeError {
    #[error("magic mismatch: expected {expected:#010X}, got {actual:#010X}, not a RAVEN index file")]
    MagicMismatch { expected: u32, actual: u32 },
    #[error("version mismatch: file version {file_version} != supported version {supported_version}, please rebuild the index")]
    VersionMismatch {
        file_version: u32,
        supported_version: u32,
    },
    #[error("crc32 mismatch: expected {expected:#010X}, got {actual:#010X}, index file corrupted")]
    CrcMismatch { expected: u32, actual: u32 },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("slice conversion error: {0}")]
    TryFromSlice(#[from] std::array::TryFromSliceError),
}

/// 索引文件头（固定 16 字节）
///
/// 设计文档 F.1 原文布局
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IndexHeader {
    /// [0..4] magic: 0x414E4E58 ("ANNX"), little-endian
    pub magic: u32,
    /// [4..8] version: u32, 格式变更时递增，当前为 1
    pub version: u32,
    /// [8..12] flags: u32, 预留扩展位
    pub flags: u32,
    /// [12..16] crc32: u32, 文件体校验和, little-endian
    pub crc32: u32,
}

impl IndexHeader {
    /// 创建新的文件头
    pub fn new(crc32: u32) -> Self {
        Self {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION,
            flags: 0,
            crc32,
        }
    }

    /// 序列化为 16 字节（little-endian）
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.flags.to_le_bytes());
        buf[12..16].copy_from_slice(&self.crc32.to_le_bytes());
        buf
    }

    /// 从 16 字节反序列化（little-endian）
    pub fn from_bytes(buf: &[u8]) -> Result<Self, SerializeError> {
        if buf.len() < HEADER_SIZE {
            return Err(SerializeError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "header too short",
            )));
        }
        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let flags = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let crc32 = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        Ok(Self { magic, version, flags, crc32 })
    }

    /// 校验文件头合法性
    ///
    /// 设计文档：
    /// - magic 不匹配 → 报错
    /// - 版本号不匹配 → 报错并提示重建索引，不静默加载
    /// - 校验和不一致 → 报错，拒绝加载
    pub fn validate(&self) -> Result<(), SerializeError> {
        if self.magic != INDEX_MAGIC {
            return Err(SerializeError::MagicMismatch {
                expected: INDEX_MAGIC,
                actual: self.magic,
            });
        }
        if self.version != INDEX_VERSION {
            return Err(SerializeError::VersionMismatch {
                file_version: self.version,
                supported_version: INDEX_VERSION,
            });
        }
        Ok(())
    }
}

/// 可序列化接口
///
/// 设计文档：索引序列化格式（含版本管理）
/// 所有需要落盘的结构实现此接口
pub trait Serializable: Sized {
    /// 序列化为字节（含文件头）
    fn serialize(&self) -> Vec<u8>;

    /// 从字节反序列化（含文件头校验）
    fn deserialize(bytes: &[u8]) -> Result<Self, SerializeError>;

    /// 序列化到文件
    fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = self.serialize();
        std::fs::write(path, bytes)
    }

    /// 从文件加载
    fn load(path: &std::path::Path) -> Result<Self, SerializeError> {
        let bytes = std::fs::read(path)?;
        Self::deserialize(&bytes)
    }
}

/// 计算字节的 CRC32 校验和
pub fn crc32(data: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(data);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let header = IndexHeader::new(0xDEADBEEF);
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);
        let restored = IndexHeader::from_bytes(&bytes).unwrap();
        assert_eq!(restored.magic, INDEX_MAGIC);
        assert_eq!(restored.version, INDEX_VERSION);
        assert_eq!(restored.crc32, 0xDEADBEEF);
    }

    #[test]
    fn header_validate_ok() {
        let header = IndexHeader::new(0);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn header_validate_magic_mismatch() {
        let mut header = IndexHeader::new(0);
        header.magic = 0x12345678;
        let err = header.validate().unwrap_err();
        assert!(matches!(err, SerializeError::MagicMismatch { .. }));
    }

    #[test]
    fn header_validate_version_mismatch() {
        let mut header = IndexHeader::new(0);
        header.version = 999;
        let err = header.validate().unwrap_err();
        assert!(matches!(err, SerializeError::VersionMismatch { .. }));
    }

    #[test]
    fn crc32_basic() {
        let data = b"hello world";
        let crc = crc32(data);
        // 已知值：crc32fast 对 "hello world" 的结果
        assert_eq!(crc, 0x0D4A1185);
    }

    #[test]
    fn header_to_bytes_little_endian() {
        let header = IndexHeader::new(0x12345678);
        let bytes = header.to_bytes();
        // magic little-endian: 58 4E 4E 41
        assert_eq!(&bytes[0..4], &[0x58, 0x4E, 0x4E, 0x41]);
        // crc32 little-endian: 78 56 34 12
        assert_eq!(&bytes[12..16], &[0x78, 0x56, 0x34, 0x12]);
    }
}
