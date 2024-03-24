import React from 'react'

const FileUpload = ({ onFileChange }) => {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    onFileChange(file)
  }
  return (
    <div>
      <input type='file' onChange={handleFileChange} />
    </div>
  )
}

export default FileUpload
