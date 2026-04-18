import time
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training and validation"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, 
                 device, num_epochs, save_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.save_path = save_path
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.training_time = 0
    
    def train_epoch(self):
        """Runs one training epoch"""

        self.model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        return running_loss / len(self.train_loader.dataset)
    
    def validate_epoch(self):
        """Runs validation"""

        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
        
        return running_loss / len(self.val_loader.dataset)
    
    def train(self):
        """Runs full training loop"""

        logger.info(f"Starting training for {self.num_epochs} epochs")
        start_time = time.time()
        
        # Reset GPU memory tracking
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f"Best model saved (val_loss: {val_loss:.4f})")
        
        self.training_time = time.time() - start_time
        logger.info(f"Training complete. Time elapsed: {self.training_time/60:.1f} min")
        
        # Capture peak GPU memory usage
        peak_vram_bytes = torch.cuda.max_memory_allocated(self.device)
        peak_vram_mb = peak_vram_bytes / (1024 ** 2)
        logger.info(f"Peak GPU Memory: {peak_vram_mb:.2f} MB")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'training_time': self.training_time,
            'peak_vram_mb': peak_vram_mb
        }
