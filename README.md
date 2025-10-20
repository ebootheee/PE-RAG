# PE-RAG: Legal Document Vector Loader for Pinecone

An interactive Python tool for uploading legal documents to Pinecone vector database with OpenAI embeddings. Designed specifically for legal document processing with proper metadata handling for n8n workflows and LangChain compatibility.

## Features

- 🔧 **Interactive Setup**: Guided configuration with smart defaults and recommendations
- 📚 **Multi-Format Support**: Process PDF, DOCX, and DOC files
- 🔑 **Secure API Management**: Safe handling of OpenAI and Pinecone API keys
- 🧪 **Connection Testing**: Verify API connections before processing
- 📊 **Smart Chunking**: Configurable text chunking with overlap for optimal retrieval
- 🎯 **LangChain Compatible**: Proper metadata structure for downstream processing
- 📈 **Progress Tracking**: Real-time upload progress with detailed logging

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[username]/PE-RAG.git
   cd PE-RAG
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install openai pinecone-client langchain langchain-openai python-docx pypdf
   ```

4. **Run the interactive loader:**
   ```bash
   python vector_loader.py
   ```

## Configuration Options

The interactive interface guides you through all necessary configurations:

### API Keys
- **OpenAI API Key**: For generating embeddings
- **Pinecone API Key**: For vector database operations

### Embedding Configuration
- **Model Selection**: 
  - `text-embedding-3-large` (3072 dimensions) - Higher accuracy
  - `text-embedding-3-small` (1536 dimensions) - Faster, cost-effective
- **Custom Dimensions**: Override default dimensions if needed

### Processing Parameters
- **Chunk Size**: Text chunk size (default: 1000 characters)
- **Chunk Overlap**: Overlap between chunks (default: 200 characters)
- **Batch Size**: Documents processed per batch (default: 10)

### Pinecone Settings
- **Index Name**: Target index for vector storage
- **Namespace**: Optional namespace for organization

## Document Processing

### Supported Formats
- **PDF**: Extracted using PyPDF2
- **DOCX**: Microsoft Word documents
- **DOC**: Legacy Word documents (converted via python-docx)

### Metadata Structure
Each document is processed with the following metadata:
```json
{
  "source": "document_filename.pdf",
  "page": 1,
  "text": "document_content_chunk",
  "chunk_id": "unique_chunk_identifier"
}
```

**Important**: Content is stored in the `text` field (not `page_content`) for proper LangChain compatibility and n8n workflow integration.

## Usage Examples

### Basic Usage
```bash
python vector_loader.py
```
Follow the interactive prompts to configure and upload your documents.

### Directory Structure
```
your_documents/
├── contract1.pdf
├── agreement2.docx
├── policy3.doc
└── subdirectory/
    └── more_docs.pdf
```

The tool will recursively process all supported documents in your specified directory.

## Advanced Configuration

### Environment Variables
You can set these environment variables to skip interactive setup:
```bash
export OPENAI_API_KEY="your_openai_key"
export PINECONE_API_KEY="your_pinecone_key"
export PINECONE_INDEX_NAME="your_index_name"
```

### Logging
All operations are logged to `vector_loader.log` with detailed information about:
- Document processing status
- Upload progress
- Error messages and troubleshooting info

## Troubleshooting

### Common Issues

1. **API Connection Failures**
   - Verify API keys are correct
   - Check internet connectivity
   - Ensure Pinecone index exists and is active

2. **Document Processing Errors**
   - Verify file formats are supported
   - Check file permissions and accessibility
   - Review logs for specific error details

3. **Memory Issues with Large Documents**
   - Reduce batch size
   - Decrease chunk size
   - Process documents in smaller batches

### Error Messages
The tool provides detailed error messages with suggested solutions. Check `vector_loader.log` for complete error traces.

## Dependencies

- `openai` - OpenAI API client for embeddings
- `pinecone-client` - Pinecone vector database client
- `langchain` - Document processing and text splitting
- `langchain-openai` - LangChain OpenAI integration
- `python-docx` - Word document processing
- `pypdf` - PDF document processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the log file for detailed error information
3. Open an issue on GitHub with:
   - Error message or unexpected behavior
   - Steps to reproduce
   - Relevant log entries

## Changelog

### v1.0.0 (2025-10-20)
- Initial release with interactive interface
- Support for PDF, DOCX, and DOC files
- OpenAI embedding integration
- Pinecone vector storage
- LangChain compatibility
- Comprehensive error handling and logging