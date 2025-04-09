#!/usr/bin/env python
"""
Test Platform Script for Brahma LLM

This script helps test the functionality of the Brahma LLM web platform,
including API endpoints, authentication, and inference capabilities.
"""
import os
import sys
import json
import time
import unittest
import requests
from typing import Dict, List, Optional, Any
import random

# Base URL for API calls
BASE_URL = "http://localhost:8000"

class BrahmaAPITester:
    """Class to test Brahma LLM API and Web platform functionality."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.token = None
        self.username = "test_user"
        self.password = "test_password"
        self.email = "test@example.com"
    
    def create_test_user(self) -> bool:
        """Create a test user for API testing."""
        try:
            url = f"{self.base_url}/register"
            data = {
                "username": self.username,
                "email": self.email,
                "password": self.password,
                "full_name": "Test User"
            }
            response = requests.post(url, json=data)
            
            if response.status_code == 201 or response.status_code == 200:
                print("✅ Test user created successfully")
                return True
            
            if response.status_code == 400 and "already exists" in response.text:
                print("✅ Test user already exists")
                return True
                
            print(f"❌ Failed to create test user: {response.status_code} - {response.text}")
            return False
        
        except Exception as e:
            print(f"❌ Error creating test user: {str(e)}")
            return False
    
    def authenticate(self) -> bool:
        """Authenticate with the API and get a token."""
        try:
            url = f"{self.base_url}/api/token"
            data = {
                "username": self.username,
                "password": self.password
            }
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                self.token = response.json().get("access_token")
                print(f"✅ Authentication successful: {self.token[:10]}...")
                return True
            
            print(f"❌ Authentication failed: {response.status_code} - {response.text}")
            return False
        
        except Exception as e:
            print(f"❌ Error during authentication: {str(e)}")
            return False
    
    def test_api_status(self) -> bool:
        """Test the API status endpoint."""
        try:
            url = f"{self.base_url}/api/status"
            response = requests.get(url)
            
            if response.status_code == 200 and response.json().get("status") == "ok":
                print("✅ API status endpoint working properly")
                return True
            
            print(f"❌ API status check failed: {response.status_code} - {response.text}")
            return False
        
        except Exception as e:
            print(f"❌ Error checking API status: {str(e)}")
            return False
    
    def test_text_generation(self) -> bool:
        """Test text generation API."""
        if not self.token:
            print("❌ Authentication required before testing text generation")
            return False
        
        try:
            url = f"{self.base_url}/api/generate"
            headers = {"Authorization": f"Bearer {self.token}"}
            data = {
                "prompt": "The future of artificial intelligence is",
                "max_new_tokens": 50,
                "temperature": 0.7,
                "do_sample": True
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                generated_text = response.json().get("generated_text", "")
                if generated_text:
                    print(f"✅ Text generation successful")
                    print(f"Generated: {generated_text[:100]}...")
                    return True
                else:
                    print("❌ Text generation returned empty response")
                    return False
            
            print(f"❌ Text generation failed: {response.status_code} - {response.text}")
            return False
        
        except Exception as e:
            print(f"❌ Error during text generation: {str(e)}")
            return False
    
    def test_chat_completion(self) -> bool:
        """Test chat completion API."""
        if not self.token:
            print("❌ Authentication required before testing chat completion")
            return False
        
        try:
            url = f"{self.base_url}/api/chat"
            headers = {"Authorization": f"Bearer {self.token}"}
            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What can you tell me about large language models?"}
                ],
                "max_new_tokens": 100,
                "temperature": 0.7
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                chat_response = response.json().get("response", "")
                if chat_response:
                    print(f"✅ Chat completion successful")
                    print(f"Response: {chat_response[:100]}...")
                    return True
                else:
                    print("❌ Chat completion returned empty response")
                    return False
            
            print(f"❌ Chat completion failed: {response.status_code} - {response.text}")
            return False
        
        except Exception as e:
            print(f"❌ Error during chat completion: {str(e)}")
            return False
    
    def test_web_pages(self) -> bool:
        """Test that web pages are accessible."""
        pages = [
            "/",  # Home
            "/login",
            "/register",
            "/dashboard",
            "/chat",
            "/playground",
            "/api-keys",
            "/account"
        ]
        
        success = True
        
        for page in pages:
            try:
                url = f"{self.base_url}{page}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    print(f"✅ Page {page} is accessible")
                else:
                    print(f"❌ Page {page} returned status code {response.status_code}")
                    success = False
            
            except Exception as e:
                print(f"❌ Error accessing page {page}: {str(e)}")
                success = False
        
        return success
    
    def test_model_management(self) -> bool:
        """Test model listing API."""
        if not self.token:
            print("❌ Authentication required before testing model management")
            return False
        
        try:
            url = f"{self.base_url}/web/models"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                models = response.json()
                print(f"✅ Model listing successful, found {len(models)} models")
                return True
            
            print(f"❌ Model listing failed: {response.status_code} - {response.text}")
            return False
        
        except Exception as e:
            print(f"❌ Error listing models: {str(e)}")
            return False
    
    def test_user_stats(self) -> bool:
        """Test user statistics API."""
        if not self.token:
            print("❌ Authentication required before testing user stats")
            return False
        
        try:
            url = f"{self.base_url}/web/user-stats"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ User stats API successful")
                print(f"Tokens generated: {stats.get('tokens_generated', 'N/A')}")
                print(f"API calls: {stats.get('api_calls', 'N/A')}")
                return True
            
            print(f"❌ User stats API failed: {response.status_code} - {response.text}")
            return False
        
        except Exception as e:
            print(f"❌ Error getting user stats: {str(e)}")
            return False
    
    def test_user_preferences(self) -> bool:
        """Test updating user preferences."""
        if not self.token:
            print("❌ Authentication required before testing user preferences")
            return False
        
        try:
            url = f"{self.base_url}/users/me/preferences"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            preferences = {
                "default_model": "brahma-7b",
                "default_temperature": 0.8,
                "use_system_theme": True,
                "stream_responses": True
            }
            
            response = requests.put(url, json=preferences, headers=headers)
            
            if response.status_code == 200:
                print(f"✅ User preferences updated successfully")
                return True
            
            print(f"❌ User preferences update failed: {response.status_code} - {response.text}")
            return False
        
        except Exception as e:
            print(f"❌ Error updating user preferences: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        results = {}
        
        print("\n🔍 STARTING BRAHMA LLM PLATFORM TESTS 🔍\n")
        
        # Test API status (no auth required)
        results["api_status"] = self.test_api_status()
        
        # Test web pages (no auth required)
        results["web_pages"] = self.test_web_pages()
        
        # Create test user
        results["create_user"] = self.create_test_user()
        
        # Authentication
        results["authentication"] = self.authenticate()
        
        if results["authentication"]:
            # Tests requiring authentication
            results["text_generation"] = self.test_text_generation()
            results["chat_completion"] = self.test_chat_completion()
            results["model_management"] = self.test_model_management()
            results["user_stats"] = self.test_user_stats()
            results["user_preferences"] = self.test_user_preferences()
        
        # Print summary
        print("\n📊 TEST RESULTS SUMMARY 📊")
        for test, passed in results.items():
            print(f"{'✅' if passed else '❌'} {test.replace('_', ' ').title()}")
        
        success_rate = sum(1 for result in results.values() if result) / len(results) * 100
        print(f"\n📈 Success Rate: {success_rate:.2f}% ({sum(1 for result in results.values() if result)}/{len(results)})")
        
        return results

def main():
    """Run the Brahma LLM platform tests."""
    tester = BrahmaAPITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
