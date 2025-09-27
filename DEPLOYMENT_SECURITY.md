# 🔐 AgriPal Deployment Security Guide

## ⚠️ CRITICAL: Database Credentials Security

**NEVER commit database credentials or API keys to public repositories!**

## 🛡️ Secure Configuration Steps

### 1. **Environment Variables in Render Dashboard**

Instead of hardcoding secrets in `render.yaml`, set them in the Render Dashboard:

1. Go to your service in Render Dashboard
2. Navigate to **Environment** tab
3. Add the following environment variables:

#### **Required Environment Variables:**
```
DATABASE_URL=postgresql+asyncpg://username:password@host:port/database_name
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
SENDGRID_API_KEY=your_sendgrid_api_key_here
SENDGRID_FROM_EMAIL=avi@aviagri.com
WEATHER_API_KEY=your_weather_api_key_here
OPENCAGE_API_KEY=your_opencage_api_key_here
```

### 2. **Update render.yaml (Already Done)**

The `render.yaml` file now uses environment variable references:
```yaml
- key: DATABASE_URL
  value: ${DATABASE_URL}
```

### 3. **Local Development Security**

For local development:
1. Copy `env.template` to `.env`
2. Fill in your local values
3. **Add `.env` to `.gitignore`**

### 4. **GitHub Repository Security**

#### **Immediate Actions Required:**
1. **Remove sensitive data from git history:**
   ```bash
   git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch render.yaml' HEAD
   git push origin --force
   ```

2. **Add to .gitignore:**
   ```
   .env
   *.env
   .env.local
   .env.production
   ```

3. **Consider making repository private** if it contains sensitive data

### 5. **Render Dashboard Configuration**

#### **Step-by-Step Setup:**

1. **Go to Render Dashboard** → Your Service → Environment
2. **Add each environment variable:**
   - Click "Add Environment Variable"
   - Enter the key (e.g., `DATABASE_URL`)
   - Enter the value (your actual credentials)
   - Click "Save Changes"

3. **Required Variables to Set:**
   - `DATABASE_URL` - Your PostgreSQL connection string
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `HF_TOKEN` - Your Hugging Face token
   - `SENDGRID_API_KEY` - Your SendGrid API key
   - `SENDGRID_FROM_EMAIL` - Your email address
   - `WEATHER_API_KEY` - Your OpenWeatherMap API key
   - `OPENCAGE_API_KEY` - Your OpenCage API key

### 6. **Security Best Practices**

#### **✅ DO:**
- Use environment variables for all secrets
- Keep `.env` files out of version control
- Use strong, unique passwords
- Rotate API keys regularly
- Monitor access logs

#### **❌ DON'T:**
- Commit secrets to git
- Share credentials in chat/email
- Use weak passwords
- Store secrets in code comments
- Use the same password for multiple services

### 7. **Verification Steps**

After setting environment variables in Render:

1. **Deploy your service**
2. **Check the logs** for:
   ```
   ✅ Database connection established successfully
   ✅ AI services initialized
   ```

3. **Test the health endpoint:**
   ```
   GET https://your-app.onrender.com/health
   ```

### 8. **Emergency Response**

If credentials are exposed:

1. **Immediately rotate all exposed keys:**
   - Change database password
   - Regenerate API keys
   - Update environment variables in Render

2. **Check for unauthorized access:**
   - Review database logs
   - Check API usage
   - Monitor for unusual activity

3. **Update security:**
   - Review access controls
   - Implement additional monitoring
   - Consider security audit

## 🎯 Summary

Your `render.yaml` is now secure and ready for public GitHub repositories. All sensitive data is moved to environment variables that you'll configure in the Render Dashboard.

**Next Steps:**
1. Set environment variables in Render Dashboard
2. Deploy your application
3. Test the database connection
4. Monitor the logs for successful startup
