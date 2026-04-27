# ngrok Setup Guide

## What is ngrok?

ngrok creates a secure tunnel to your local API, providing a public HTTPS URL that you can use to access your API from anywhere on the internet. This is perfect for testing your API with external websites or mobile apps.

## Installation

### macOS (using Homebrew)
```bash
brew install ngrok/ngrok/ngrok
```

### Or download directly
1. Visit https://ngrok.com/download
2. Download for your platform
3. Extract and add to PATH

### Sign up (Free)
1. Go to https://dashboard.ngrok.com/signup
2. Create a free account
3. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken

## Setup

### 1. Authenticate
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN
```

### 2. Start your API locally
```bash
cd copol_prediction/api
python app.py
```

The API should be running on `http://localhost:8000`

### 3. Start ngrok tunnel
In a new terminal:
```bash
ngrok http 8000
```

You'll see output like:
```
Forwarding   https://abc123.ngrok-free.app -> http://localhost:8000
```

### 4. Use the HTTPS URL
Use the `https://` URL shown by ngrok in your web applications:

```javascript
// Example: Use the ngrok URL instead of localhost
const API_URL = 'https://abc123.ngrok-free.app';

fetch(`${API_URL}/preprocess_all`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        monomer1_smiles: "C=CC1=CC=CC=C1",
        monomer2_smiles: "C=C(C)C(=O)OCCO",
        solvent_smiles: "CCO",
        method: "solvent",
        polytype: "free radical",
        temperature: 60.0
    })
})
.then(res => res.json())
.then(data => console.log(data));
```

## Advanced Usage

### Static Domain (Paid)
If you have a paid ngrok account, you can use a static domain:
```bash
ngrok http 8000 --domain=your-static-domain.ngrok.io
```

### Custom Subdomain (Paid)
```bash
ngrok http 8000 --subdomain=my-api
```

### With Authentication
```bash
ngrok http 8000 --basic-auth="username:password"
```

## Important Notes

1. **Free tier limitations:**
   - URL changes every time you restart ngrok
   - Limited requests per minute
   - Connection timeout after inactivity

2. **For production:**
   - Use a proper hosting service (AWS, Heroku, etc.)
   - Or use ngrok paid plan for static domains

3. **Security:**
   - ngrok URLs are public - anyone with the URL can access your API
   - Consider adding authentication for production use

## Troubleshooting

### ngrok not found
Make sure ngrok is in your PATH or use full path:
```bash
/path/to/ngrok http 8000
```

### Port already in use
If port 8000 is already in use, change the API port or use ngrok's port forwarding:
```bash
ngrok http 8001  # Forward to different port
```

### Connection refused
Make sure your API is running on localhost:8000 before starting ngrok.

## Example Integration

Use the ngrok URL as the base URL in any HTTP client.
